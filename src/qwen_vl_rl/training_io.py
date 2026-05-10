from __future__ import annotations

import json
import math
from pathlib import Path
import re
import warnings


CHECKPOINT_DIR_PATTERN = re.compile(r'checkpoint-(\d+)$')


def append_metric(output_dir: Path, record: dict) -> None:
    metrics_path = output_dir / 'metrics.jsonl'
    with metrics_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def prepare_checkpoint_dir(output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / f'checkpoint-{step}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def checkpoint_step(checkpoint_dir: str | Path) -> int | None:
    match = CHECKPOINT_DIR_PATTERN.fullmatch(Path(checkpoint_dir).name)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    output_path = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for child in output_path.glob('checkpoint-*'):
        step = checkpoint_step(child)
        if step is not None and child.is_dir():
            candidates.append((step, child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_resume_checkpoint(
    resume_from_checkpoint: str | None,
    output_dir: str | Path,
    project_root: str | Path | None = None,
) -> Path | None:
    if not resume_from_checkpoint:
        return None

    output_path = Path(output_dir)
    resume_value = resume_from_checkpoint.strip()
    if resume_value.lower() in {'latest', 'last'}:
        latest = find_latest_checkpoint(output_path)
        if latest is None:
            raise FileNotFoundError(
                f'No checkpoint-* directory found under {output_path}'
            )
        checkpoint_dir = latest
    else:
        raw_path = Path(resume_value).expanduser()
        candidates: list[Path]
        if raw_path.is_absolute():
            candidates = [raw_path]
        else:
            candidates = [output_path / raw_path]
            if project_root is not None:
                candidates.append(Path(project_root) / raw_path)
            candidates.append(Path.cwd() / raw_path)

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique_candidates.append(candidate)
                seen.add(key)

        checkpoint_dir = next((candidate for candidate in unique_candidates if candidate.exists()), None)
        if checkpoint_dir is None:
            checked = ', '.join(str(candidate) for candidate in unique_candidates)
            raise FileNotFoundError(
                f'Resume checkpoint {resume_from_checkpoint!r} was not found. Checked: {checked}'
            )

    if checkpoint_dir.name == 'adapter':
        checkpoint_dir = checkpoint_dir.parent
    if not checkpoint_dir.is_dir():
        raise ValueError(f'Resume checkpoint is not a directory: {checkpoint_dir}')
    if not (checkpoint_dir / 'adapter').is_dir():
        raise ValueError(
            f'Resume checkpoint must contain an adapter/ directory: {checkpoint_dir}'
        )
    return checkpoint_dir.resolve()


def load_training_state(checkpoint_dir: str | Path) -> dict:
    state_path = Path(checkpoint_dir) / 'training_state.json'
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding='utf-8'))


def resume_step_from_checkpoint(checkpoint_dir: str | Path) -> int:
    training_state = load_training_state(checkpoint_dir)
    if 'step' in training_state:
        return int(training_state['step'])
    return checkpoint_step(checkpoint_dir) or 0


def save_optimizer_and_training_state(
    optimizer,
    checkpoint_dir: Path,
    training_state: dict,
    scheduler=None,
) -> None:
    import torch

    torch.save(optimizer.state_dict(), checkpoint_dir / 'optimizer.pt')
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / 'scheduler.pt')
    (checkpoint_dir / 'training_state.json').write_text(
        json.dumps(training_state, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def load_optimizer_state_if_available(
    optimizer,
    checkpoint_dir: str | Path,
    map_location: str = 'cpu',
) -> bool:
    import torch

    optimizer_path = Path(checkpoint_dir) / 'optimizer.pt'
    if not optimizer_path.exists():
        return False
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
    return True


def load_scheduler_state_if_available(
    scheduler,
    checkpoint_dir: str | Path,
    map_location: str = 'cpu',
) -> bool:
    import torch

    scheduler_path = Path(checkpoint_dir) / 'scheduler.pt'
    if not scheduler_path.exists():
        return False
    scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
    return True


def advance_scheduler_to_step(scheduler, target_step: int) -> int:
    if target_step <= 0:
        return 0
    inner_scheduler = getattr(scheduler, 'scheduler', scheduler)
    current_step = int(getattr(inner_scheduler, 'last_epoch', 0))
    steps_to_advance = max(int(target_step) - current_step, 0)
    if steps_to_advance == 0:
        return 0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`',
        )
        for _ in range(steps_to_advance):
            inner_scheduler.step()
    return steps_to_advance


def estimate_total_training_steps(
    num_batches: int,
    num_train_epochs: int,
    num_processes: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int | None = None,
) -> int:
    # 这里的 num_batches 传入的是“全局数据集按单卡 batch_size 切分后的 batch 数”。
    # 在多卡 DDP 下，每个进程只会消费其中大约 1 / num_processes 的 batch，
    # 因此这里先换算为单进程视角下的局部 batch 数。
    #
    # 注意：Accelerate 在 dataloader 结束时会强制同步最后一个“不满梯度累积步数”
    # 的尾 batch，因此每个 epoch 的实际 optimizer step 数应按 ceil 计算，
    # 且这个 ceil 需要在每个 epoch 内单独生效，而不是把所有 epoch 的 batch
    # 先合并后再整除。
    batches_per_process = math.ceil(num_batches / max(num_processes, 1))
    updates_per_epoch = math.ceil(
        batches_per_process / max(gradient_accumulation_steps, 1)
    )
    total_steps = max(
        1,
        updates_per_epoch * num_train_epochs,
    )
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    return total_steps


def log_metrics(
    accelerator,
    prefix: str,
    metrics: dict[str, float],
    total_steps: int | None = None,
) -> None:
    if accelerator.is_main_process:
        step = metrics.get('step')
        pieces = []
        if step is not None and total_steps is not None:
            pieces.append(f"step={int(step)}/{total_steps}")
        for key, value in metrics.items():
            if key in {'step', 'epoch', 'total_steps'}:
                continue
            if isinstance(value, (int, float)):
                pieces.append(f'{key}={value:.4f}')
        print(f'[{prefix}] ' + ' '.join(pieces), flush=True)
