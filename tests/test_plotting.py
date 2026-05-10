import json
from pathlib import Path

import pytest

from src.qwen_vl_rl.plotting import infer_run_kind, load_metric_records, render_metrics_curve


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        ''.join(json.dumps(record) + '\n' for record in records),
        encoding='utf-8',
    )


def test_infer_run_kind_detects_sft_and_rl_metrics():
    assert infer_run_kind([{'phase': 'eval', 'eval_exact_match': 0.5}]) == 'sft'
    assert infer_run_kind([{'phase': 'train', 'reward_mean': 0.5, 'value_loss': 0.1}]) == 'ppo'
    assert infer_run_kind([{'phase': 'train', 'reward_mean': 0.5, 'advantage_abs_mean': 0.2}]) == 'grpo'


def test_render_metrics_curve_accepts_run_directory(tmp_path: Path):
    records = [
        {'phase': 'train', 'step': 1, 'reward_mean': 1.0, 'accuracy': 1.0, 'valid_option_rate': 1.0, 'kl_mean': 0.1, 'response_length_mean': 3.0, 'loss': 0.2},
        {'phase': 'train', 'step': 2, 'reward_mean': -0.25, 'accuracy': 0.0, 'valid_option_rate': 1.0, 'kl_mean': 0.2, 'response_length_mean': 3.0, 'loss': 0.3},
        {'phase': 'eval', 'step': 2, 'reward_mean': 0.4, 'accuracy': 0.6, 'valid_option_rate': 1.0, 'response_length_mean': 2.0},
    ]
    _write_jsonl(tmp_path / 'metrics.jsonl', records)

    output = render_metrics_curve(tmp_path, kind='ppo', rolling_window=2)

    assert output == tmp_path / 'training_curve.png'
    assert output.exists()
    assert output.stat().st_size > 0


def test_load_metric_records_rejects_empty_metrics_file(tmp_path: Path):
    metrics_path = tmp_path / 'metrics.jsonl'
    metrics_path.write_text('', encoding='utf-8')

    with pytest.raises(ValueError, match='No metric records'):
        load_metric_records(metrics_path)
