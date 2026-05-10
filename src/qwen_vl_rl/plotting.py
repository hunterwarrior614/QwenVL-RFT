from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


RunKind = Literal['auto', 'sft', 'ppo', 'grpo']


def resolve_metrics_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_dir():
        target = target / 'metrics.jsonl'
    if not target.exists():
        raise FileNotFoundError(f'Metrics file not found: {target}')
    return target


def load_metric_records(metrics_path: str | Path) -> list[dict[str, Any]]:
    path = resolve_metrics_path(metrics_path)
    records: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path}:{line_number}: {exc}') from exc
    if not records:
        raise ValueError(f'No metric records found in {path}')
    return records


def infer_run_kind(records: list[dict[str, Any]], kind: RunKind = 'auto') -> Literal['sft', 'ppo', 'grpo']:
    if kind != 'auto':
        return kind

    keys = {key for record in records for key in record}
    if 'eval_exact_match' in keys or ('loss' in keys and 'reward_mean' not in keys):
        return 'sft'
    if 'advantage_abs_mean' in keys or 'kl_loss' in keys:
        return 'grpo'
    if 'value_loss' in keys or 'reward_mean' in keys:
        return 'ppo'
    return 'sft'


def render_metrics_curve(
    metrics_path: str | Path,
    output_path: str | Path | None = None,
    kind: RunKind = 'auto',
    rolling_window: int = 1,
    dpi: int = 160,
) -> Path:
    path = resolve_metrics_path(metrics_path)
    records = load_metric_records(path)
    run_kind = infer_run_kind(records, kind)
    output = Path(output_path) if output_path else path.parent / 'training_curve.png'
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    if run_kind == 'sft':
        fig = _render_sft(records, rolling_window)
    else:
        fig = _render_rl(records, run_kind.upper(), rolling_window)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _render_sft(records: list[dict[str, Any]], rolling_window: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_metric(
        axes[0],
        records,
        train_metric='loss',
        eval_metric='eval_loss',
        title='SFT Loss',
        ylabel='Loss',
        rolling_window=rolling_window,
    )
    _plot_metric(
        axes[1],
        records,
        train_metric=None,
        eval_metric='eval_exact_match',
        title='SFT Eval Exact Match',
        ylabel='Accuracy',
        rolling_window=rolling_window,
        y_limits=(0.0, 1.0),
    )
    return fig


def _render_rl(records: list[dict[str, Any]], title_prefix: str, rolling_window: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    flat_axes = axes.flatten()

    _plot_metric(
        flat_axes[0],
        records,
        train_metric='reward_mean',
        eval_metric='reward_mean',
        title=f'{title_prefix} Reward',
        ylabel='Reward',
        rolling_window=rolling_window,
    )
    _plot_metric(
        flat_axes[1],
        records,
        train_metric='accuracy',
        eval_metric='accuracy',
        title=f'{title_prefix} Accuracy',
        ylabel='Accuracy',
        rolling_window=rolling_window,
        y_limits=(0.0, 1.0),
    )
    _plot_metric(
        flat_axes[2],
        records,
        train_metric='valid_option_rate',
        eval_metric='valid_option_rate',
        title=f'{title_prefix} Valid Option Rate',
        ylabel='Rate',
        rolling_window=rolling_window,
        y_limits=(0.0, 1.0),
    )
    _plot_metric(
        flat_axes[3],
        records,
        train_metric='kl_mean',
        eval_metric=None,
        title=f'{title_prefix} KL',
        ylabel='KL',
        rolling_window=rolling_window,
    )
    _plot_metric(
        flat_axes[4],
        records,
        train_metric='response_length_mean',
        eval_metric='response_length_mean',
        title=f'{title_prefix} Response Length',
        ylabel='Tokens',
        rolling_window=rolling_window,
    )
    _plot_available_metrics(
        flat_axes[5],
        records,
        metric_names=['loss', 'policy_loss', 'value_loss', 'kl_loss', 'entropy'],
        title=f'{title_prefix} Loss Terms',
        rolling_window=rolling_window,
    )
    return fig


def _plot_metric(
    ax,
    records: list[dict[str, Any]],
    train_metric: str | None,
    eval_metric: str | None,
    title: str,
    ylabel: str,
    rolling_window: int,
    y_limits: tuple[float, float] | None = None,
) -> None:
    plotted = False
    if train_metric:
        train_steps, train_values = _series(records, train_metric, phase='train')
        if train_steps:
            train_values = _rolling_mean(train_values, rolling_window)
            ax.plot(train_steps, train_values, label=f'train_{train_metric}', linewidth=1.8)
            plotted = True

    if eval_metric:
        eval_steps, eval_values = _series(records, eval_metric, phase='eval')
        if eval_steps:
            ax.plot(eval_steps, eval_values, label=f'eval_{eval_metric}', linewidth=2.0, marker='o')
            plotted = True

    _finish_axis(ax, title, ylabel, plotted, y_limits)


def _plot_available_metrics(
    ax,
    records: list[dict[str, Any]],
    metric_names: list[str],
    title: str,
    rolling_window: int,
) -> None:
    plotted = False
    for metric_name in metric_names:
        steps, values = _series(records, metric_name, phase='train')
        if not steps:
            continue
        values = _rolling_mean(values, rolling_window)
        ax.plot(steps, values, label=metric_name, linewidth=1.6)
        plotted = True
    _finish_axis(ax, title, 'Value', plotted)


def _series(
    records: list[dict[str, Any]],
    metric_name: str,
    phase: str | None = None,
) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    for index, record in enumerate(records, start=1):
        if phase is not None and record.get('phase') != phase:
            continue
        value = record.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        step = record.get('step', index)
        if not isinstance(step, (int, float)):
            step = index
        steps.append(float(step))
        values.append(float(value))
    return steps, values


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    output: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        denom = min(index + 1, window)
        output.append(running_sum / denom)
    return output


def _finish_axis(
    ax,
    title: str,
    ylabel: str,
    plotted: bool,
    y_limits: tuple[float, float] | None = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
