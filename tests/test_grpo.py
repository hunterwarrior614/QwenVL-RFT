import torch

from src.qwen_vl_rl.grpo import compute_group_advantages


def test_compute_group_advantages_centers_rewards_per_prompt():
    scores = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    advantages = compute_group_advantages(
        scores=scores,
        batch_size=2,
        group_size=3,
        whiten=False,
    )

    expected = torch.tensor([1 / 3, -2 / 3, 1 / 3, 0.0, 0.0, 0.0])
    assert torch.allclose(advantages, expected)


def test_compute_group_advantages_whitens_nonconstant_groups():
    scores = torch.tensor([1.0, 0.0, 1.0])

    advantages = compute_group_advantages(
        scores=scores,
        batch_size=1,
        group_size=3,
        whiten=True,
    )

    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_compute_group_advantages_returns_zero_for_eval_single_generation():
    scores = torch.tensor([1.0, 0.0])

    advantages = compute_group_advantages(
        scores=scores,
        batch_size=2,
        group_size=1,
        whiten=True,
    )

    assert torch.equal(advantages, torch.zeros_like(scores))
