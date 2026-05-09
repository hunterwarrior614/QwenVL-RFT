import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.qwen_vl_rl.training_io import (
    append_metric,
    estimate_total_training_steps,
    prepare_checkpoint_dir,
    save_optimizer_and_training_state,
)
from src.qwen_vl_rl.utils import resolve_config_paths_in_dict, resolve_object_paths


def test_append_metric_appends_jsonl_record(tmp_path: Path):
    append_metric(tmp_path, {'phase': 'train', 'step': 1, 'loss': 0.5})
    append_metric(tmp_path, {'phase': 'eval', 'step': 1, 'accuracy': 1.0})

    metrics_path = tmp_path / 'metrics.jsonl'
    lines = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines()]

    assert lines == [
        {'phase': 'train', 'step': 1, 'loss': 0.5},
        {'phase': 'eval', 'step': 1, 'accuracy': 1.0},
    ]


def test_prepare_checkpoint_dir_and_save_training_state(tmp_path: Path):
    checkpoint_dir = prepare_checkpoint_dir(tmp_path, step=3)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)

    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={'step': 3, 'algorithm': 'ppo'},
    )

    assert checkpoint_dir == tmp_path / 'checkpoint-3'
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / 'optimizer.pt').exists()
    assert json.loads((checkpoint_dir / 'training_state.json').read_text(encoding='utf-8')) == {
        'step': 3,
        'algorithm': 'ppo',
    }


def test_resolve_config_paths_in_dict_resolves_required_keys(tmp_path: Path):
    project_root = tmp_path
    config = {
        'output_dir': 'outputs/sft/default',
        'base_model_name_or_path': '../Qwen/Qwen2.5-VL-3B-Instruct',
        'train_file': 'data/train.jsonl',
    }

    resolved = resolve_config_paths_in_dict(
        config,
        project_root,
        required_keys=['output_dir', 'base_model_name_or_path', 'train_file'],
    )

    assert resolved['output_dir'] == str(project_root / 'outputs/sft/default')
    assert resolved['base_model_name_or_path'] == str(project_root / '../Qwen/Qwen2.5-VL-3B-Instruct')
    assert resolved['train_file'] == str(project_root / 'data/train.jsonl')


def test_resolve_object_paths_handles_required_and_optional_attrs(tmp_path: Path):
    project_root = tmp_path
    obj = SimpleNamespace(
        train_file='data/train.jsonl',
        base_model_name_or_path='../Qwen/model',
        sft_adapter_path=None,
        output_dir='outputs/ppo/default',
    )

    resolve_object_paths(
        obj,
        project_root,
        required_attrs=['train_file', 'base_model_name_or_path', 'output_dir'],
        optional_attrs=['sft_adapter_path'],
    )

    assert obj.train_file == str(project_root / 'data/train.jsonl')
    assert obj.base_model_name_or_path == str(project_root / '../Qwen/model')
    assert obj.output_dir == str(project_root / 'outputs/ppo/default')
    assert obj.sft_adapter_path is None


def test_estimate_total_training_steps_accounts_for_processes_and_accumulation():
    assert estimate_total_training_steps(
        num_batches=1000,
        num_train_epochs=2,
        num_processes=4,
        gradient_accumulation_steps=4,
    ) == 125

    assert estimate_total_training_steps(
        num_batches=1000,
        num_train_epochs=2,
        num_processes=4,
        gradient_accumulation_steps=1,
    ) == 500

    assert estimate_total_training_steps(
        num_batches=1000,
        num_train_epochs=2,
        num_processes=4,
        gradient_accumulation_steps=4,
        max_steps=20,
    ) == 20
