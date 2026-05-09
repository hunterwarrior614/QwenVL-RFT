from __future__ import annotations

import json
from pathlib import Path


def append_metric(output_dir: Path, record: dict) -> None:
    metrics_path = output_dir / 'metrics.jsonl'
    with metrics_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def prepare_checkpoint_dir(output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / f'checkpoint-{step}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_optimizer_and_training_state(
    optimizer,
    checkpoint_dir: Path,
    training_state: dict,
) -> None:
    import torch

    torch.save(optimizer.state_dict(), checkpoint_dir / 'optimizer.pt')
    (checkpoint_dir / 'training_state.json').write_text(
        json.dumps(training_state, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


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
