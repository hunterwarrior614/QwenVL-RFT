from types import SimpleNamespace

import torch
import pytest

from scripts.train.train_grpo_qwen_vl_lora import (
    run_evaluation as run_grpo_evaluation,
    summarize_rollout as summarize_grpo_rollout,
)
from scripts.train.train_ppo_qwen_vl_lora import (
    run_evaluation as run_ppo_evaluation,
    summarize_rollout as summarize_ppo_rollout,
)


def _build_rollout():
    return SimpleNamespace(
        response_mask=torch.ones((3, 2), dtype=torch.bool),
        pred_letters=['A', 'B', None],
        answer_keys=['A', 'C', 'D'],
        scores=torch.tensor([1.0, -0.25, -0.5]),
        old_logprobs=torch.zeros((3, 2)),
        ref_logprobs=torch.zeros((3, 2)),
        advantages=torch.tensor([0.5, -0.5, 0.0]),
    )


def test_ppo_summary_separates_reward_mean_from_accuracy():
    metrics = summarize_ppo_rollout(_build_rollout())

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_grpo_summary_separates_reward_mean_from_accuracy():
    metrics = summarize_grpo_rollout(_build_rollout())

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_ppo_evaluation_uses_total_count_as_denominator(monkeypatch):
    rollout = _build_rollout()

    def fake_generate_rollout_batch(**kwargs):
        return rollout

    import scripts.train.train_ppo_qwen_vl_lora as ppo_train

    monkeypatch.setattr(ppo_train, 'generate_rollout_batch', fake_generate_rollout_batch)

    metrics = run_ppo_evaluation(
        policy=_FakePolicy(),
        reference_model=None,
        processor=None,
        valid_loader=[{}],
        config=SimpleNamespace(generation=None, ppo=None),
        accelerator=_FakeAccelerator(),
    )

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_grpo_evaluation_uses_total_count_as_denominator(monkeypatch):
    rollout = _build_rollout()

    def fake_generate_grpo_rollout_batch(**kwargs):
        return rollout

    import scripts.train.train_grpo_qwen_vl_lora as grpo_train

    monkeypatch.setattr(grpo_train, 'generate_grpo_rollout_batch', fake_generate_grpo_rollout_batch)

    metrics = run_grpo_evaluation(
        policy=_FakePolicy(),
        reference_model=None,
        processor=None,
        valid_loader=[{}],
        config=SimpleNamespace(generation=None, grpo=None),
        accelerator=_FakeAccelerator(),
    )

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


class _FakePolicy:
    def eval(self):
        pass

    def train(self):
        pass


class _FakeAccelerator:
    device = torch.device('cpu')
    num_processes = 1
