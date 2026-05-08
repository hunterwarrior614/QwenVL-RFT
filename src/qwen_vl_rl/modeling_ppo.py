from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f'Unsupported torch dtype: {dtype_name}')
    return mapping[dtype_name]


def build_quantization_config(model_config) -> BitsAndBytesConfig | None:
    if not model_config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=get_torch_dtype(model_config.bnb_4bit_compute_dtype),
    )


# 为 PPO 训练设计的策略网络包装类
# 它将语言模型（作为策略网络 Actor）与一个价值头（Value Head） 组合在一起，用于估计状态价值函数
class PPOPolicyWithValueHead(nn.Module):
    def __init__(self, policy_model: nn.Module, value_head_dropout: float = 0.0):
        super().__init__()

        # 原始语言模型（可以是基础模型或经过 LoRA 微调的模型），负责生成 logits 和隐藏状态：
        # 这里 logits 是原始模型输出的隐藏状态经过 LM head（通常是线性层） 输出的结果，用于计算 Token 预测概率；
        # 这里提取出隐藏状态是用于估算每一个 Token 位的状态价值（value head）
        self.policy_model = policy_model

        hidden_size = _get_hidden_size(policy_model.config)
        value_dtype = _get_model_dtype(policy_model)

        # 在价值头之前添加 dropout 用于正则化
        self.value_dropout = nn.Dropout(value_head_dropout)

        # 一个线性层，将隐藏状态映射到价值
        self.value_head = nn.Linear(hidden_size, 1, dtype=value_dtype)

        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    @property
    def config(self):
        return self.policy_model.config

    def generate(self, *args, **kwargs):
        return self.policy_model.generate(*args, **kwargs)

    def forward(self, **kwargs):
        return self.policy_model(**kwargs)

    def compute_value_from_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.value_dropout(hidden_states)).squeeze(-1)

    def evaluate_actions(self, input_ids: torch.Tensor, **model_kwargs) -> dict[str, torch.Tensor]:
        outputs = self.policy_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **model_kwargs,
        )
        return compute_policy_outputs_from_model_outputs(
            model_wrapper=self,
            input_ids=input_ids,
            outputs=outputs,
        )


def build_policy_model(model_config, lora_config) -> PPOPolicyWithValueHead:
    base_model = _load_base_model(model_config)

    if model_config.load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    if model_config.sft_adapter_path:
        # 从一个已经训练好的 PEFT（参数高效微调）适配器文件（例如 LoRA 权重）中加载参数，
        # 并将其应用到基础模型 base_model 上，同时允许这个适配器在后续训练中继续被更新（即可训练）
        policy_backbone = PeftModel.from_pretrained(
            base_model,
            model_config.sft_adapter_path,
            is_trainable=True,
        )
    else:
        target_modules = _resolve_lora_target_modules(base_model, lora_config.target_modules_regex)
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            target_modules=target_modules,
            task_type='CAUSAL_LM',
        )
        policy_backbone = get_peft_model(base_model, peft_config)

    # 用于降低显存占用的技术参数，开启后会在前向传播时不保存所有中间激活值，
    # 而是在反向传播时重新计算需要的部分，从而用计算时间换取显存空间
    if model_config.gradient_checkpointing:
        policy_backbone.gradient_checkpointing_enable() # 启用检查点
        policy_backbone.config.use_cache = False        # 禁用 KV 缓存（因为检查点与缓存不兼容）
        policy_backbone.enable_input_require_grads()    # 确保输入需要梯度（防止某些层丢失梯度）

    return PPOPolicyWithValueHead(policy_backbone)


def build_reference_model(model_config) -> nn.Module:
    reference_model = _load_base_model(model_config)
    if model_config.sft_adapter_path:
        reference_model = PeftModel.from_pretrained(
            reference_model,
            model_config.sft_adapter_path,
            is_trainable=False,
        )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


def save_policy_checkpoint(policy: PPOPolicyWithValueHead, output_dir: str | Path, metadata: dict[str, Any]) -> None:
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.policy_model.save_pretrained(save_dir / 'adapter')
    torch.save(policy.value_head.state_dict(), save_dir / 'value_head.pt')
    with (save_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def categorical_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def compute_policy_outputs_from_model_outputs(
    model_wrapper: PPOPolicyWithValueHead,
    input_ids: torch.Tensor,
    outputs,
) -> dict[str, torch.Tensor]:
    logits = outputs.logits[:, :-1, :]
    hidden_states = outputs.hidden_states[-1][:, :-1, :]
    values = model_wrapper.compute_value_from_hidden_states(hidden_states)
    target_tokens = input_ids[:, 1:]
    logprobs = gather_log_probs(logits, target_tokens)
    entropy = categorical_entropy_from_logits(logits)
    return {
        'logprobs': logprobs,
        'values': values,
        'entropy': entropy,
    }


def _load_base_model(model_config) -> Qwen2_5_VLForConditionalGeneration:
    kwargs: dict[str, Any] = {
        # 是否允许执行从模型仓库（Model Hub）下载的自定义 Python 代码
        'trust_remote_code': model_config.trust_remote_code,
        'dtype': get_torch_dtype(model_config.torch_dtype),
    }
    quantization_config = build_quantization_config(model_config)
    if quantization_config is not None:
        kwargs['quantization_config'] = quantization_config
    if model_config.attn_implementation:
        kwargs['attn_implementation'] = model_config.attn_implementation
    if model_config.use_flash_attention_2:
        kwargs['attn_implementation'] = 'flash_attention_2'
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config.base_model_name_or_path,
        **kwargs,
    )


def _get_hidden_size(config) -> int:
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return int(config.text_config.hidden_size)
    if hasattr(config, 'hidden_size'):
        return int(config.hidden_size)
    raise ValueError('Could not infer hidden size for value head')


# 获取模型参数的浮点数据类型
# 通过遍历模型的所有参数，找到第一个浮点类型的参数，并返回其 dtype
def _get_model_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _resolve_lora_target_modules(model: nn.Module, target_regex: str) -> list[str]:
    matched = _match_module_names(model, target_regex)
    if not matched and '\\\\' in target_regex:
        matched = _match_module_names(model, bytes(target_regex, 'utf-8').decode('unicode_escape'))
    if not matched:
        raise ValueError(f'No LoRA target modules matched regex: {target_regex}')
    return matched


def _match_module_names(model: nn.Module, target_regex: str) -> list[str]:
    pattern = re.compile(target_regex)
    return [name for name, _ in model.named_modules() if pattern.fullmatch(name)]
