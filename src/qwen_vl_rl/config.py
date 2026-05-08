from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_file: str = ''
    train_size: int = 1000
    eval_size: int = 121
    test_size: int = 121
    split_seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_workers: int = 0
    image_max_longest_edge: int | None = None


@dataclass
class ModelConfig:
    base_model_name_or_path: str = ''
    sft_adapter_path: str | None = None
    torch_dtype: str = 'bfloat16'
    attn_implementation: str = 'sdpa'
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    use_flash_attention_2: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = 'bfloat16'


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = 'none'
    target_modules_regex: str = (
        r'.*language_model\\.layers\\.\\d+\\.'
        r'(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$'
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int = 8
    min_new_tokens: int = 1
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    eval_max_new_tokens: int = 8


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class PPOConfig:
    # 在一轮迭代中，用同一个策略前向传播生成的一批 rollout 数量
    per_device_prompt_batch_size: int = 1  

    # 在生成的一批 rollout 中，利用这些 rollout 进行多轮 PPO 更新，一轮迭代中划分的每个 minibatch 的 rollout 数量
    # 值一般小于 per_device_prompt_batch_size
    per_device_minibatch_size: int = 1  

    # 在生成的一批 rollout 中，利用这些 rollout 进行多轮 PPO 更新的轮数
    ppo_epochs: int = 2
    cliprange: float = 0.2
    value_cliprange: float = 0.2
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    whiten_advantages: bool = True
    entropy_coef: float = 0.0


@dataclass
class GRPOConfig:
    per_device_prompt_batch_size: int = 1
    num_generations: int = 4
    per_device_minibatch_size: int = 1
    grpo_epochs: int = 1
    cliprange: float = 0.2
    kl_coef: float = 0.02
    whiten_group_advantages: bool = True
    entropy_coef: float = 0.0


@dataclass
class LoggingConfig:
    output_dir: str = 'outputs/ppo/default'
    logging_steps: int = 1
    eval_steps: int = 25
    save_steps: int = 50
    save_total_limit: int = 3


@dataclass
class PPOTrainConfig:
    seed: int = 42
    num_train_epochs: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


@dataclass
class GRPOTrainConfig:
    seed: int = 42
    num_train_epochs: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value

# 递归地将字典中的键值对更新到 dataclass 实例的对应字段上。
# 如果某个字段本身也是一个 dataclass，且字典中对应值是嵌套字典，则继续递归更新该子 dataclass。
def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path) -> PPOTrainConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    config = PPOTrainConfig(
        data=DataConfig(train_file=''),
        model=ModelConfig(base_model_name_or_path=''),
    )
    return _update_dataclass(config, payload or {})


def load_grpo_config(config_path: str | Path) -> GRPOTrainConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    config = GRPOTrainConfig(
        data=DataConfig(train_file=''),
        model=ModelConfig(base_model_name_or_path=''),
        logging=LoggingConfig(output_dir='outputs/grpo/default'),
    )
    return _update_dataclass(config, payload or {})
