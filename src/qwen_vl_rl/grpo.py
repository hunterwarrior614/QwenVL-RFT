from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .modeling_ppo import categorical_entropy_from_logits, gather_log_probs
from .ppo import (
    build_response_attention_mask,
    decode_response_texts,
    masked_mean,
    move_batch_to_device,
    slice_visual_features,
    _normalize_eos_ids,
)
from .reward import score_choice_predictions


@dataclass
class GRPORolloutBatch:
    sequences: torch.Tensor
    full_attention_mask: torch.Tensor
    response_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    advantages: torch.Tensor
    scores: torch.Tensor
    response_texts: list[str]
    pred_letters: list[str | None]
    answer_keys: list[str]
    sample_ids: list[int]
    prompt_padded_length: int
    visual_patch_counts: torch.Tensor
    group_size: int


@torch.no_grad()
def generate_grpo_rollout_batch(
    policy,
    reference_model,
    processor,
    batch: dict[str, Any],
    generation_config,
    grpo_config,
    accelerator,
    eval_mode: bool = False,
) -> GRPORolloutBatch:
    prompt_inputs = move_batch_to_device(batch['prompt_inputs'], accelerator.device)
    prompt_attention_mask = prompt_inputs['attention_mask']
    prompt_padded_length = prompt_inputs['input_ids'].shape[1]
    group_size = 1 if eval_mode else grpo_config.num_generations
    original_batch_size = prompt_inputs['input_ids'].shape[0]

    repeated_prompt_inputs = _repeat_prompt_inputs(prompt_inputs, group_size)
    repeated_attention_mask = repeated_prompt_inputs['attention_mask']
    visual_patch_counts = repeated_prompt_inputs['image_grid_thw'].prod(dim=1)

    generation_kwargs = {
        'input_ids': repeated_prompt_inputs['input_ids'],
        'attention_mask': repeated_attention_mask,
        'pixel_values': repeated_prompt_inputs['pixel_values'],
        'image_grid_thw': repeated_prompt_inputs['image_grid_thw'],
        'do_sample': False if eval_mode else generation_config.do_sample,
        'max_new_tokens': (
            generation_config.eval_max_new_tokens
            if eval_mode
            else generation_config.max_new_tokens
        ),
        'min_new_tokens': generation_config.min_new_tokens,
        'repetition_penalty': generation_config.repetition_penalty,
        'pad_token_id': processor.tokenizer.pad_token_id,
        'eos_token_id': processor.tokenizer.eos_token_id,
        'return_dict_in_generate': True,
        'output_scores': False,
        'use_cache': True,
    }
    if not eval_mode and generation_config.do_sample:
        generation_kwargs['temperature'] = generation_config.temperature
        generation_kwargs['top_p'] = generation_config.top_p
    if not eval_mode and generation_config.top_k and generation_config.top_k > 0:
        generation_kwargs['top_k'] = generation_config.top_k

    unwrapped_policy = accelerator.unwrap_model(policy)
    policy_was_training = unwrapped_policy.training
    unwrapped_policy.eval()
    sequences = unwrapped_policy.generate(**generation_kwargs).sequences

    eos_token_ids = _normalize_eos_ids(processor.tokenizer.eos_token_id)
    response_attention_mask = build_response_attention_mask(
        sequences[:, prompt_padded_length:],
        eos_token_ids,
    )
    full_attention_mask = torch.cat([repeated_attention_mask, response_attention_mask], dim=1)
    response_mask = torch.cat(
        [torch.zeros_like(repeated_attention_mask, dtype=torch.bool), response_attention_mask.bool()],
        dim=1,
    )
    shifted_response_mask = response_mask[:, 1:]

    model_inputs = {
        'attention_mask': full_attention_mask,
        'pixel_values': repeated_prompt_inputs['pixel_values'],
        'image_grid_thw': repeated_prompt_inputs['image_grid_thw'],
    }
    policy_outputs = unwrapped_policy(
        input_ids=sequences,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **model_inputs,
    )
    old_logprobs = gather_log_probs(policy_outputs.logits[:, :-1, :], sequences[:, 1:])
    if policy_was_training:
        unwrapped_policy.train()

    ref_outputs = reference_model(
        input_ids=sequences,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **model_inputs,
    )
    ref_logprobs = gather_log_probs(ref_outputs.logits[:, :-1, :], sequences[:, 1:])

    response_texts = decode_response_texts(
        processor=processor,
        sequences=sequences,
        response_attention_mask=response_attention_mask,
        prompt_padded_length=prompt_padded_length,
    )
    answer_keys = [
        answer_key
        for answer_key in batch['answer_keys']
        for _ in range(group_size)
    ]
    reward_output = score_choice_predictions(response_texts, answer_keys)
    scores = torch.tensor(reward_output['rewards'], device=accelerator.device, dtype=torch.float32)
    advantages = compute_group_advantages(
        scores=scores,
        batch_size=original_batch_size,
        group_size=group_size,
        whiten=grpo_config.whiten_group_advantages,
    )

    return GRPORolloutBatch(
        sequences=sequences.detach().cpu(),
        full_attention_mask=full_attention_mask.detach().cpu(),
        response_mask=shifted_response_mask.detach().cpu(),
        pixel_values=repeated_prompt_inputs['pixel_values'].detach().cpu(),
        image_grid_thw=repeated_prompt_inputs['image_grid_thw'].detach().cpu(),
        old_logprobs=old_logprobs.detach().cpu(),
        ref_logprobs=ref_logprobs.detach().cpu(),
        advantages=advantages.detach().cpu(),
        scores=scores.detach().cpu(),
        response_texts=response_texts,
        pred_letters=reward_output['pred_letters'],
        answer_keys=answer_keys,
        sample_ids=[
            sample_id
            for sample_id in batch['sample_ids']
            for _ in range(group_size)
        ],
        prompt_padded_length=prompt_padded_length,
        visual_patch_counts=visual_patch_counts.detach().cpu(),
        group_size=group_size,
    )


def compute_grpo_losses(
    policy,
    minibatch: dict[str, torch.Tensor],
    cliprange: float,
    kl_coef: float,
    entropy_coef: float,
) -> dict[str, torch.Tensor]:
    outputs = policy(
        input_ids=minibatch['sequences'],
        attention_mask=minibatch['full_attention_mask'],
        pixel_values=minibatch['pixel_values'],
        image_grid_thw=minibatch['image_grid_thw'],
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    logprobs = gather_log_probs(logits, minibatch['sequences'][:, 1:])
    entropy = categorical_entropy_from_logits(logits)

    mask = minibatch['response_mask']
    old_logprobs = minibatch['old_logprobs']
    ref_logprobs = minibatch['ref_logprobs']
    advantages = minibatch['advantages'].unsqueeze(1)

    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss_1 = -advantages * ratio
    pg_loss_2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    policy_loss = masked_mean(torch.maximum(pg_loss_1, pg_loss_2), mask)

    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1.0
    kl_loss = masked_mean(kl, mask)
    entropy_loss = masked_mean(entropy, mask)
    total_loss = policy_loss + kl_coef * kl_loss - entropy_coef * entropy_loss

    clipfrac = masked_mean((torch.abs(ratio - 1.0) > cliprange).float(), mask)
    approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
    ref_kl = masked_mean(logprobs - ref_logprobs, mask)

    return {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'kl_loss': kl_loss,
        'entropy': entropy_loss,
        'clipfrac': clipfrac,
        'approx_kl': approx_kl,
        'ref_kl': ref_kl,
    }


def build_grpo_minibatch(
    rollout: GRPORolloutBatch,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cpu_indices = indices.detach().cpu()
    return {
        'sequences': rollout.sequences[cpu_indices].to(device, non_blocking=True),
        'full_attention_mask': rollout.full_attention_mask[cpu_indices].to(device, non_blocking=True),
        'response_mask': rollout.response_mask[cpu_indices].to(device, non_blocking=True),
        'pixel_values': slice_visual_features(
            rollout.pixel_values,
            rollout.visual_patch_counts,
            cpu_indices,
        ).to(device, non_blocking=True),
        'image_grid_thw': rollout.image_grid_thw[cpu_indices].to(device, non_blocking=True),
        'old_logprobs': rollout.old_logprobs[cpu_indices].to(device, non_blocking=True),
        'ref_logprobs': rollout.ref_logprobs[cpu_indices].to(device, non_blocking=True),
        'advantages': rollout.advantages[cpu_indices].to(device, non_blocking=True),
    }


def compute_group_advantages(
    scores: torch.Tensor,
    batch_size: int,
    group_size: int,
    whiten: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    grouped = scores.view(batch_size, group_size)
    if group_size == 1:
        return torch.zeros_like(scores)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if whiten:
        std = grouped.std(dim=1, unbiased=False, keepdim=True)
        centered = centered / torch.clamp(std, min=eps)
    return centered.reshape(-1)


def _repeat_prompt_inputs(prompt_inputs: dict[str, Any], group_size: int) -> dict[str, Any]:
    if group_size == 1:
        return prompt_inputs

    repeated: dict[str, Any] = {}
    image_grid_thw = prompt_inputs['image_grid_thw']
    patch_counts = image_grid_thw.prod(dim=1)
    offsets = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, device=patch_counts.device, dtype=patch_counts.dtype),
                patch_counts,
            ]
        ),
        dim=0,
    )

    for key, value in prompt_inputs.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue
        if key == 'pixel_values':
            chunks = []
            for row_index in range(image_grid_thw.shape[0]):
                start = int(offsets[row_index].item())
                end = int(offsets[row_index + 1].item())
                for _ in range(group_size):
                    chunks.append(value[start:end])
            repeated[key] = torch.cat(chunks, dim=0)
        elif key == 'image_grid_thw':
            repeated[key] = value.repeat_interleave(group_size, dim=0)
        elif value.shape[0] == image_grid_thw.shape[0]:
            repeated[key] = value.repeat_interleave(group_size, dim=0)
        else:
            repeated[key] = value
    return repeated
