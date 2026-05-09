from types import SimpleNamespace

import torch
import torch.nn as nn

from src.qwen_vl_rl.modeling_common import (
    build_quantization_config_from_fields,
    get_torch_dtype,
    match_module_names,
    resolve_lora_target_modules,
)


def test_get_torch_dtype_supports_expected_aliases():
    assert get_torch_dtype('float16') == torch.float16
    assert get_torch_dtype('fp16') == torch.float16
    assert get_torch_dtype('bfloat16') == torch.bfloat16
    assert get_torch_dtype('bf16') == torch.bfloat16
    assert get_torch_dtype('float32') == torch.float32
    assert get_torch_dtype('fp32') == torch.float32


def test_build_quantization_config_from_fields_supports_dict_input():
    config = build_quantization_config_from_fields(
        {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': 'bfloat16',
        }
    )

    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == 'nf4'
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == torch.bfloat16


def test_build_quantization_config_from_fields_supports_attribute_input():
    config = build_quantization_config_from_fields(
        SimpleNamespace(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype='float16',
        )
    )

    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == 'nf4'
    assert config.bnb_4bit_use_double_quant is False
    assert config.bnb_4bit_compute_dtype == torch.float16


def test_build_quantization_config_from_fields_returns_none_when_disabled():
    assert build_quantization_config_from_fields({'load_in_4bit': False}) is None


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Module()])
        self.language_model.layers[0].self_attn = nn.Module()
        self.language_model.layers[0].self_attn.q_proj = nn.Linear(2, 2)
        self.language_model.layers[0].self_attn.k_proj = nn.Linear(2, 2)


def test_match_module_names_returns_matching_names_without_validation():
    model = TinyModule()

    matched = match_module_names(
        model,
        r'.*language_model\.layers\.0\.self_attn\.(q_proj|k_proj)$',
    )

    assert matched == [
        'language_model.layers.0.self_attn.q_proj',
        'language_model.layers.0.self_attn.k_proj',
    ]


def test_resolve_lora_target_modules_supports_unicode_escaped_regex():
    model = TinyModule()

    matched = resolve_lora_target_modules(
        model,
        r'.*language_model\\.layers\\.0\\.self_attn\\.(q_proj|k_proj)$',
    )

    assert matched == [
        'language_model.layers.0.self_attn.q_proj',
        'language_model.layers.0.self_attn.k_proj',
    ]
