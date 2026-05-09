from __future__ import annotations

import re

import torch
from transformers import BitsAndBytesConfig


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


def build_quantization_config_from_fields(config_like) -> BitsAndBytesConfig | None:
    load_in_4bit = (
        config_like.get('load_in_4bit', False)
        if isinstance(config_like, dict)
        else getattr(config_like, 'load_in_4bit', False)
    )
    if not load_in_4bit:
        return None

    bnb_4bit_quant_type = (
        config_like['bnb_4bit_quant_type']
        if isinstance(config_like, dict)
        else config_like.bnb_4bit_quant_type
    )
    bnb_4bit_use_double_quant = (
        config_like['bnb_4bit_use_double_quant']
        if isinstance(config_like, dict)
        else config_like.bnb_4bit_use_double_quant
    )
    bnb_4bit_compute_dtype = (
        config_like['bnb_4bit_compute_dtype']
        if isinstance(config_like, dict)
        else config_like.bnb_4bit_compute_dtype
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=get_torch_dtype(bnb_4bit_compute_dtype),
    )


def match_module_names(model, target_regex: str) -> list[str]:
    pattern = re.compile(target_regex)
    return [name for name, _ in model.named_modules() if pattern.fullmatch(name)]


def resolve_lora_target_modules(model, target_regex: str) -> list[str]:
    matched = match_module_names(model, target_regex)
    if not matched and '\\' in target_regex:
        matched = match_module_names(model, bytes(target_regex, 'utf-8').decode('unicode_escape'))
    if not matched:
        raise ValueError(f'No LoRA target modules matched regex: {target_regex}')
    return matched
