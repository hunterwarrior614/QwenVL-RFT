from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .modeling_ppo import compute_policy_outputs_from_model_outputs, gather_log_probs
from .reward import score_choice_predictions


# 封装一次 PPO rollout 生成的所有中间数据
@dataclass
class RolloutBatch:
    # 完整的 token 序列，包括 prompt tokens（输入问题/图像占位符）和 生成的 response tokens（模型回答）
    # 形状为 (batch_size, max_sequence_length)，其中 max_sequence_length = prompt_padded_length + max_response_length
    # 一个 sequence 中每个 token 对应一个整数 ID（即 tokenizer 词汇表中的索引），例如 101 表示某个词或子词，
    # 这些整数 ID 是模型理解文本的输入形式，模型会将其映射为嵌入向量再进行计算。
    sequences: torch.Tensor

    # 对应 sequences 的“填充注意力掩码”，1 表示有效 token，0 表示 padding，形状同 sequences
    # 它由 prompt 部分的 mask（来自 processor）和 response 部分的 mask（根据 EOS 截断生成）拼接而成
    #
    # PS：在 Hugging Face Transformers 以及大多数现代 Transformer 实现中，注意力掩码被分为两个部分：
    #       1. 填充掩码：标记哪些位置是真实的 token（值=1），哪些位置是填充的无效 token（值=0）
    #       2. 因果掩码：对于自回归生成（语言模型），确保 token i 只能 attend 到 token 0..i
    # 在模型内部，这两种掩码会被结合起来，形成一个真正的注意力矩阵
    full_attention_mask: torch.Tensor

    # 布尔掩码，形状同 sequences
    # 仅对 response 部分的 token 标记为 1（“需要计算损失和优势的位置”），prompt 部分和对齐的 padding 均为 0
    response_mask: torch.Tensor

    # 仅 prompt 部分的注意力掩码（来自 processor 的原始输出），形状为 (batch_size, prompt_padded_length)
    prompt_attention_mask: torch.Tensor

    # 经过 processor 处理后的图像张量（归一化、resize、切块等），作为视觉语言模型的视觉输入
    pixel_values: torch.Tensor

    # 描述每个图像被切分成多少个块（patch）以及时间维度（通常 1）
    # 形状为 (num_images, 3)，每行 [temporal, height, width] 表示图像块网格大小（Qwen-VL 特有）
    image_grid_thw: torch.Tensor

    # 旧策略下每个 response token 的对数概率，形状为 (batch_size, max_sequence_length - 1)（shifted）
    old_logprobs: torch.Tensor

    # 旧策略的价值网络对每个 response token 位置的状态价值估计 V(s)，形状同上
    old_values: torch.Tensor

    # 参考模型（固定，通常是 SFT 模型）对同一 response token 的对数概率，形状同上
    ref_logprobs: torch.Tensor

    # 使用 GAE 计算的优势估计值 A(s,a)，形状同上
    advantages: torch.Tensor

    # 折扣回报 G_t = A_t + V(s_t)，也即目标价值，用于价值网络（Critic）的回归目标。形状同上
    returns: torch.Tensor

    # 每个 token 位置的即时奖励，作为 GAE 计算的原始奖励序列。形状同上
    # 对于 response 中的非最后一个 token，奖励来自 KL 惩罚项 -kl_coef * (old_logprobs - ref_logprobs)；
    # 对于最后一个 response token，额外的最终奖励（环境奖励，如 VQA 得分）会加到该位置上。
    rewards: torch.Tensor

    # 每个样本的环境奖励（例如 VQA 准确率，1 或 0，或更细粒度的得分），
    # 赋予每一个 response 最后一个有效 token，即 rewards 中的额外最终奖励
    scores: torch.Tensor

    # 解码后的模型回答文本（去除了特殊 token 和 prompt 部分）
    response_texts: list[str]

    # 从 response_texts 中提取出的选项字母（如 "A"、"B"），若无法解析则为 None
    pred_letters: list[str | None]

    # 每个样本的正确答案字母（来自数据集），即 pred_letters 的期望结果
    answer_keys: list[str]

    # 数据集中每个样本的唯一标识符，与 response_texts 对应
    sample_ids: list[int]

    # 在批次内，所有 prompt 经过 padding 后的统一长度
    prompt_padded_length: int

    # 每个样本包含的图像块数量（image_grid_thw.prod(dim=1) 的结果）
    visual_patch_counts: torch.Tensor


@torch.no_grad()
def generate_rollout_batch(
    policy,
    reference_model,
    processor,
    batch: dict[str, Any],
    generation_config,
    ppo_config,
    accelerator,
) -> RolloutBatch:
    prompt_inputs = move_batch_to_device(batch['prompt_inputs'], accelerator.device)
    prompt_attention_mask = prompt_inputs['attention_mask']
    prompt_padded_length = prompt_inputs['input_ids'].shape[1]
    visual_patch_counts = prompt_inputs['image_grid_thw'].prod(dim=1)

    generation_kwargs = {
        'input_ids': prompt_inputs['input_ids'],
        'attention_mask': prompt_attention_mask,
        'pixel_values': prompt_inputs['pixel_values'],
        'image_grid_thw': prompt_inputs['image_grid_thw'],
        'do_sample': generation_config.do_sample,
        'temperature': generation_config.temperature,
        'top_p': generation_config.top_p,
        'max_new_tokens': generation_config.max_new_tokens,
        'min_new_tokens': generation_config.min_new_tokens,
        'repetition_penalty': generation_config.repetition_penalty,
        'pad_token_id': processor.tokenizer.pad_token_id,
        'eos_token_id': processor.tokenizer.eos_token_id,
        'return_dict_in_generate': True,
        'output_scores': False,
        'use_cache': True,
    }
    if generation_config.top_k and generation_config.top_k > 0:
        generation_kwargs['top_k'] = generation_config.top_k

    unwrapped_policy = accelerator.unwrap_model(policy)

    # 布尔属性 training，保存当前模型（policy）的训练模式状态，
    # 确保模型在生成 rollout 数据后，能够恢复到正确的模式继续后续的 PPO 训练
    policy_was_training = unwrapped_policy.training

    unwrapped_policy.eval()
    sequences = unwrapped_policy.generate(**generation_kwargs).sequences

    if policy_was_training:
        unwrapped_policy.train()

    eos_token_ids = _normalize_eos_ids(processor.tokenizer.eos_token_id)
    response_attention_mask = build_response_attention_mask(sequences[:, prompt_padded_length:], eos_token_ids)
    full_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    response_mask = torch.cat(
        [torch.zeros_like(prompt_attention_mask, dtype=torch.bool), response_attention_mask.bool()],
        dim=1,
    )

    model_inputs = {
        'attention_mask': full_attention_mask,
        'pixel_values': prompt_inputs['pixel_values'],
        'image_grid_thw': prompt_inputs['image_grid_thw'],
    }

    old_policy_outputs = unwrapped_policy.evaluate_actions(sequences, **model_inputs)

    ref_outputs = reference_model(
        input_ids=sequences,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **model_inputs,
    )
    ref_logprobs = gather_log_probs(ref_outputs.logits[:, :-1, :], sequences[:, 1:])

    response_texts = decode_response_texts(
        processor=processor,
        sequences=sequences,
        response_attention_mask=response_attention_mask,
        prompt_padded_length=prompt_padded_length,
    )
    reward_output = score_choice_predictions(response_texts, batch['answer_keys'])
    scores = torch.tensor(reward_output['rewards'], device=accelerator.device, dtype=torch.float32)

    old_logprobs = old_policy_outputs['logprobs']
    old_values = old_policy_outputs['values']
    ref_logprobs = ref_logprobs
    shifted_response_mask = response_mask[:, 1:]

    token_rewards = -ppo_config.kl_coef * (old_logprobs - ref_logprobs)
    token_rewards = token_rewards * shifted_response_mask
    for row_index in range(token_rewards.shape[0]):
        valid_positions = torch.nonzero(shifted_response_mask[row_index], as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        token_rewards[row_index, valid_positions[-1]] += scores[row_index]

    advantages, returns = compute_gae(
        rewards=token_rewards,
        values=old_values,
        mask=shifted_response_mask,
        gamma=ppo_config.gamma,
        lam=ppo_config.lam,
    )

    return RolloutBatch(
        sequences=sequences.detach().cpu(),
        full_attention_mask=full_attention_mask.detach().cpu(),
        response_mask=shifted_response_mask.detach().cpu(),
        prompt_attention_mask=prompt_attention_mask.detach().cpu(),
        pixel_values=prompt_inputs['pixel_values'].detach().cpu(),
        image_grid_thw=prompt_inputs['image_grid_thw'].detach().cpu(),
        old_logprobs=old_logprobs.detach().cpu(),
        old_values=old_values.detach().cpu(),
        ref_logprobs=ref_logprobs.detach().cpu(),
        advantages=advantages.detach().cpu(),
        returns=returns.detach().cpu(),
        rewards=token_rewards.detach().cpu(),
        scores=scores.detach().cpu(),
        response_texts=response_texts,
        pred_letters=reward_output['pred_letters'],
        answer_keys=batch['answer_keys'],
        sample_ids=batch['sample_ids'],
        prompt_padded_length=prompt_padded_length,
        visual_patch_counts=visual_patch_counts.detach().cpu(),
    )


def compute_ppo_losses(policy, minibatch: dict[str, torch.Tensor], cliprange: float, value_cliprange: float, vf_coef: float, entropy_coef: float) -> dict[str, torch.Tensor]:
    model_outputs = policy(
        input_ids=minibatch['sequences'],
        attention_mask=minibatch['full_attention_mask'],
        pixel_values=minibatch['pixel_values'],
        image_grid_thw=minibatch['image_grid_thw'],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    policy_module = policy.module if hasattr(policy, 'module') else policy
    outputs = compute_policy_outputs_from_model_outputs(
        model_wrapper=policy_module,
        input_ids=minibatch['sequences'],
        outputs=model_outputs,
    )
    logprobs = outputs['logprobs']
    values = outputs['values']
    entropy = outputs['entropy']

    mask = minibatch['response_mask']
    old_logprobs = minibatch['old_logprobs']
    old_values = minibatch['old_values']
    advantages = minibatch['advantages']
    returns = minibatch['returns']

    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss_1 = -advantages * ratio
    pg_loss_2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    policy_loss = masked_mean(torch.maximum(pg_loss_1, pg_loss_2), mask)

    value_pred_clipped = old_values + torch.clamp(values - old_values, -value_cliprange, value_cliprange)
    value_loss_1 = (values - returns) ** 2
    value_loss_2 = (value_pred_clipped - returns) ** 2
    value_loss = 0.5 * masked_mean(torch.maximum(value_loss_1, value_loss_2), mask)

    entropy_loss = masked_mean(entropy, mask)
    total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy_loss

    clipfrac = masked_mean((torch.abs(ratio - 1.0) > cliprange).float(), mask)
    approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)

    return {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy_loss,
        'clipfrac': clipfrac,
        'approx_kl': approx_kl,
    }


def build_minibatch(
    rollout: RolloutBatch,
    indices: torch.Tensor,
    whiten_advantages: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cpu_indices = indices.detach().cpu()
    advantages = rollout.advantages[cpu_indices]
    response_mask = rollout.response_mask[cpu_indices]
    if whiten_advantages:
        advantages = masked_whiten(advantages, response_mask)

    batch = {
        'sequences': rollout.sequences[cpu_indices].to(device, non_blocking=True),
        'full_attention_mask': rollout.full_attention_mask[cpu_indices].to(device, non_blocking=True),
        'response_mask': response_mask.to(device, non_blocking=True),
        'pixel_values': slice_visual_features(rollout.pixel_values, rollout.visual_patch_counts, cpu_indices).to(device, non_blocking=True),
        'image_grid_thw': rollout.image_grid_thw[cpu_indices].to(device, non_blocking=True),
        'old_logprobs': rollout.old_logprobs[cpu_indices].to(device, non_blocking=True),
        'old_values': rollout.old_values[cpu_indices].to(device, non_blocking=True),
        'advantages': advantages,
        'returns': rollout.returns[cpu_indices].to(device, non_blocking=True),
    }
    batch['advantages'] = batch['advantages'].to(device, non_blocking=True)
    return batch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    mean = (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    var = (((values - mean) ** 2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    whitened = (values - mean) / torch.sqrt(var + eps)
    return whitened * mask_f


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for row_index in range(rewards.shape[0]):
        valid_positions = torch.nonzero(mask[row_index], as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        last_advantage = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        next_value = torch.tensor(0.0, device=values.device, dtype=values.dtype)
        for position in reversed(valid_positions.tolist()):
            delta = rewards[row_index, position] + gamma * next_value - values[row_index, position]
            last_advantage = delta + gamma * lam * last_advantage
            advantages[row_index, position] = last_advantage
            next_value = values[row_index, position]
        returns[row_index, valid_positions] = advantages[row_index, valid_positions] + values[row_index, valid_positions]
    return advantages, returns


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def build_response_attention_mask(generated_tokens: torch.Tensor, eos_token_ids: set[int]) -> torch.Tensor:
    batch_size, max_steps = generated_tokens.shape
    attention_mask = torch.zeros((batch_size, max_steps), dtype=torch.long, device=generated_tokens.device)
    for row_index in range(batch_size):
        tokens = generated_tokens[row_index].tolist()
        response_length = max_steps
        for token_index, token_id in enumerate(tokens):
            if token_id in eos_token_ids:
                response_length = token_index + 1
                break
        attention_mask[row_index, :response_length] = 1
    return attention_mask


def decode_response_texts(processor, sequences: torch.Tensor, response_attention_mask: torch.Tensor, prompt_padded_length: int) -> list[str]:
    texts: list[str] = []
    for row_index in range(sequences.shape[0]):
        response_tokens = sequences[row_index, prompt_padded_length:]
        response_length = int(response_attention_mask[row_index].sum().item())
        trimmed = response_tokens[:response_length].tolist()
        texts.append(processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip())
    return texts


def slice_visual_features(pixel_values: torch.Tensor, visual_patch_counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    offsets = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, device=visual_patch_counts.device, dtype=visual_patch_counts.dtype),
                visual_patch_counts,
            ]
        ),
        dim=0,
    )
    chunks = []
    for raw_index in indices.tolist():
        start = int(offsets[raw_index].item())
        end = int(offsets[raw_index + 1].item())
        chunks.append(pixel_values[start:end])
    return torch.cat(chunks, dim=0)


# 将可能为整数或整数列表的 eos_token_id 统一标准化为一个整数集合，
# 方便后续构建 response_attention_mask 时判断生成的 token 是否为 EOS
def _normalize_eos_ids(eos_token_id: int | list[int]) -> set[int]:
    if isinstance(eos_token_id, list):
        return set(int(item) for item in eos_token_id)
    return {int(eos_token_id)}
