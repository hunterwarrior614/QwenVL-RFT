import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.qwen_vl_rl.training_io import (
    advance_scheduler_to_step,
    append_metric,
    checkpoint_step,
    estimate_total_training_steps,
    find_latest_checkpoint,
    load_optimizer_state_if_available,
    load_scheduler_state_if_available,
    load_training_state,
    prepare_checkpoint_dir,
    resolve_resume_checkpoint,
    resume_step_from_checkpoint,
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={'step': 3, 'algorithm': 'ppo'},
        scheduler=scheduler,
    )

    assert checkpoint_dir == tmp_path / 'checkpoint-3'
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / 'optimizer.pt').exists()
    assert (checkpoint_dir / 'scheduler.pt').exists()
    assert json.loads((checkpoint_dir / 'training_state.json').read_text(encoding='utf-8')) == {
        'step': 3,
        'algorithm': 'ppo',
    }
    assert load_training_state(checkpoint_dir) == {'step': 3, 'algorithm': 'ppo'}
    assert resume_step_from_checkpoint(checkpoint_dir) == 3


def test_checkpoint_resolution_supports_latest_explicit_and_adapter_paths(tmp_path: Path):
    old_checkpoint = prepare_checkpoint_dir(tmp_path, step=1)
    new_checkpoint = prepare_checkpoint_dir(tmp_path, step=12)
    (old_checkpoint / 'adapter').mkdir()
    (new_checkpoint / 'adapter').mkdir()

    assert checkpoint_step(new_checkpoint) == 12
    assert find_latest_checkpoint(tmp_path) == new_checkpoint
    assert resolve_resume_checkpoint('latest', tmp_path) == new_checkpoint.resolve()
    assert resolve_resume_checkpoint('checkpoint-12', tmp_path) == new_checkpoint.resolve()
    assert resolve_resume_checkpoint(
        str(new_checkpoint / 'adapter'),
        tmp_path,
    ) == new_checkpoint.resolve()


def test_checkpoint_resolution_handles_project_relative_path(tmp_path: Path):
    project_root = tmp_path / 'project'
    checkpoint = project_root / 'outputs' / 'ppo' / 'run' / 'checkpoint-7'
    (checkpoint / 'adapter').mkdir(parents=True)

    resolved = resolve_resume_checkpoint(
        'outputs/ppo/run/checkpoint-7',
        output_dir=tmp_path / 'other',
        project_root=project_root,
    )

    assert resolved == checkpoint.resolve()


def test_resume_step_falls_back_to_checkpoint_directory_name(tmp_path: Path):
    checkpoint = prepare_checkpoint_dir(tmp_path, step=9)
    (checkpoint / 'adapter').mkdir()

    assert resume_step_from_checkpoint(checkpoint) == 9


def test_load_optimizer_and_scheduler_state_if_available(tmp_path: Path):
    checkpoint_dir = prepare_checkpoint_dir(tmp_path, step=4)
    source_param = torch.nn.Parameter(torch.tensor(1.0))
    source_optimizer = torch.optim.Adam([source_param], lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(
        source_optimizer,
        lr_lambda=lambda step: 1.0 / (step + 1),
    )
    source_param.grad = torch.tensor(0.5)
    source_optimizer.step()
    source_scheduler.step()
    save_optimizer_and_training_state(
        optimizer=source_optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={'step': 4},
        scheduler=source_scheduler,
    )

    target_param = torch.nn.Parameter(torch.tensor(1.0))
    target_optimizer = torch.optim.Adam([target_param], lr=0.1)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer,
        lr_lambda=lambda step: 1.0 / (step + 1),
    )

    assert load_optimizer_state_if_available(target_optimizer, checkpoint_dir)
    assert load_scheduler_state_if_available(target_scheduler, checkpoint_dir)
    assert target_scheduler.last_epoch == source_scheduler.last_epoch


def test_advance_scheduler_to_step_fast_forwards_when_state_file_is_missing():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 / (step + 1),
    )

    assert advance_scheduler_to_step(scheduler, 4) == 4
    assert scheduler.last_epoch == 4
    assert advance_scheduler_to_step(scheduler, 2) == 0
    assert scheduler.last_epoch == 4


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
    ) == 126

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


def test_estimate_total_training_steps_keeps_partial_accumulation_tail_per_epoch():
    assert estimate_total_training_steps(
        num_batches=1001,
        num_train_epochs=1,
        num_processes=1,
        gradient_accumulation_steps=4,
    ) == 251

    assert estimate_total_training_steps(
        num_batches=1000,
        num_train_epochs=1,
        num_processes=4,
        gradient_accumulation_steps=4,
    ) == 63

    assert estimate_total_training_steps(
        num_batches=1000,
        num_train_epochs=2,
        num_processes=4,
        gradient_accumulation_steps=4,
    ) == 126
