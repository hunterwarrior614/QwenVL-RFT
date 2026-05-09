from types import SimpleNamespace

import torch

from scripts.train.train_grpo_qwen_vl_lora import summarize_rollout as summarize_grpo_rollout
from scripts.train.train_ppo_qwen_vl_lora import summarize_rollout as summarize_ppo_rollout


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

    assert metrics['reward_mean'] == torch.tensor([1.0, -0.25, -0.5]).mean().item()
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_grpo_summary_separates_reward_mean_from_accuracy():
    metrics = summarize_grpo_rollout(_build_rollout())

    assert metrics['reward_mean'] == torch.tensor([1.0, -0.25, -0.5]).mean().item()
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3
