#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen_vl_rl.config import load_config
from src.qwen_vl_rl.data import QwenVLPPOCollator, create_split_datasets
from src.qwen_vl_rl.modeling_ppo import build_policy_model, build_reference_model, save_policy_checkpoint
from src.qwen_vl_rl.ppo import build_minibatch, compute_ppo_losses, generate_rollout_batch
from src.qwen_vl_rl.utils import dump_json, ensure_dir, resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL PPO with LoRA on Thyme VQA data.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ppo_qwen_vl_lora.yaml',
    )
    parser.add_argument('--max-steps', type=int, default=None, help='Optional cap on PPO prompt updates.')
    parser.add_argument('--eval-only', action='store_true')
    return parser.parse_args()


def summarize_rollout(rollout) -> dict[str, float]:
    response_lengths = rollout.response_mask.sum(dim=1).float()
    valid_option_rate = sum(
        letter is not None for letter in rollout.pred_letters
    ) / max(len(rollout.pred_letters), 1)
    approx_kl = (
        (rollout.old_logprobs - rollout.ref_logprobs) * rollout.response_mask
    ).sum() / rollout.response_mask.sum().clamp_min(1)
    return {
        'reward_mean': float(rollout.scores.mean().item()),
        'accuracy': float(rollout.scores.mean().item()),
        'valid_option_rate': float(valid_option_rate),
        'response_length_mean': float(response_lengths.mean().item()),
        'kl_mean': float(approx_kl.item()),
    }


@torch.no_grad()
def run_evaluation(
    policy,
    reference_model,
    processor,
    eval_loader,
    config,
    accelerator,
    max_batches: int | None = None,
) -> dict[str, float]:
    policy.eval()
    rewards = []
    lengths = []
    valid = 0
    total = 0

    for batch_index, batch in enumerate(eval_loader):
        rollout = generate_rollout_batch(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            batch=batch,
            generation_config=config.generation,
            ppo_config=config.ppo,
            accelerator=accelerator,
        )
        rewards.append(float(rollout.scores.mean().item()))
        lengths.append(float(rollout.response_mask.sum(dim=1).float().mean().item()))
        valid += sum(letter is not None for letter in rollout.pred_letters)
        total += len(rollout.pred_letters)
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    policy.train()
    reward_mean = sum(rewards) / max(len(rewards), 1)
    length_mean = sum(lengths) / max(len(lengths), 1)
    return {
        'reward_mean': reward_mean,
        'accuracy': reward_mean,
        'valid_option_rate': valid / max(total, 1),
        'response_length_mean': length_mean,
    }


def append_metric(output_dir: Path, record: dict) -> None:
    metrics_path = output_dir / 'metrics.jsonl'
    with metrics_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def render_training_curve(output_dir: Path) -> None:
    metrics_path = output_dir / 'metrics.jsonl'
    if not metrics_path.exists():
        return

    records = []
    with metrics_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return

    train_steps = [item['step'] for item in records if item['phase'] == 'train']
    train_rewards = [
        item.get('reward_mean', 0.0) for item in records if item['phase'] == 'train'
    ]
    train_accuracies = [
        item.get('accuracy', 0.0) for item in records if item['phase'] == 'train'
    ]
    train_kls = [
        item.get('kl_mean', 0.0) for item in records if item['phase'] == 'train'
    ]

    eval_steps = [item['step'] for item in records if item['phase'] == 'eval']
    eval_rewards = [
        item.get('reward_mean', 0.0) for item in records if item['phase'] == 'eval'
    ]
    eval_accuracies = [
        item.get('accuracy', 0.0) for item in records if item['phase'] == 'eval'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        train_steps, train_rewards, label='train_reward', color='#1f77b4', linewidth=2
    )
    if eval_steps:
        axes[0].plot(
            eval_steps,
            eval_rewards,
            label='eval_reward',
            color='#d62728',
            linewidth=2,
            marker='o',
        )
    axes[0].set_title('PPO Reward')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        train_steps,
        train_accuracies,
        label='train_accuracy',
        color='#2ca02c',
        linewidth=2,
    )
    if eval_steps:
        axes[1].plot(
            eval_steps,
            eval_accuracies,
            label='eval_accuracy',
            color='#ff7f0e',
            linewidth=2,
            marker='o',
        )
    axes[1].set_title('PPO Accuracy')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(train_steps, train_kls, label='train_kl', color='#9467bd', linewidth=2)
    axes[2].set_title('PPO KL')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('KL')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / 'training_curve.png', dpi=160)
    plt.close(fig)


def save_checkpoint(policy, optimizer, output_dir: Path, step: int, config) -> None:
    unwrapped = policy
    if hasattr(policy, 'module'):
        unwrapped = policy.module
    checkpoint_dir = output_dir / f'checkpoint-{step}'
    save_policy_checkpoint(
        policy=unwrapped,
        output_dir=checkpoint_dir,
        metadata={
            'step': step,
            'base_model': config.model.base_model_name_or_path,
        },
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / 'optimizer.pt')
    (checkpoint_dir / 'training_state.json').write_text(
        json.dumps(
            {
                'step': step,
                'base_model': config.model.base_model_name_or_path,
                'output_dir': str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


def log_metrics(
    accelerator: Accelerator,
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.data.train_file = str(resolve_project_path(config.data.train_file, PROJECT_ROOT))
    config.model.base_model_name_or_path = str(
        resolve_project_path(config.model.base_model_name_or_path, PROJECT_ROOT)
    )
    config.model.sft_adapter_path = (
        str(resolve_project_path(config.model.sft_adapter_path, PROJECT_ROOT))
        if config.model.sft_adapter_path
        else None
    )
    config.logging.output_dir = str(resolve_project_path(config.logging.output_dir, PROJECT_ROOT))
    set_seed(config.seed)

    output_dir = ensure_dir(config.logging.output_dir)
    dump_json(config.to_dict(), output_dir / 'resolved_config.json')
    metrics_path = output_dir / 'metrics.jsonl'
    metrics_path.write_text('', encoding='utf-8')   # 清空文件内容

    # 创建了一个 Accelerator 对象
    # 其作用是自动处理分布式训练（多 GPU、TPU、混合精度等），同时简化设备管理、梯度累积和混合精度训练等代码
    accelerator = Accelerator(gradient_accumulation_steps=1)

    if accelerator.is_main_process:
        print('Loading processor and datasets...')

    # 创建了一个多模态处理器（Processor），专门用于处理视觉语言模型（如 Qwen2.5-VL）的输入
    processor = AutoProcessor.from_pretrained(config.model.base_model_name_or_path)

    train_dataset, eval_dataset, test_dataset = create_split_datasets(
        jsonl_path=config.data.train_file,
        train_size=config.data.train_size,
        eval_size=config.data.eval_size,
        test_size=config.data.test_size,
        split_seed=config.data.split_seed,
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples,
    )
    collator = QwenVLPPOCollator(
        processor,
        image_max_longest_edge=config.data.image_max_longest_edge,
    )
    dump_json(
        {
            'train_size': len(train_dataset),
            'eval_size': len(eval_dataset),
            'test_size': len(test_dataset),
            'split_seed': config.data.split_seed,
            'train_ids': [sample['sample_id'] for sample in train_dataset],
            'eval_ids': [sample['sample_id'] for sample in eval_dataset],
            'test_ids': [sample['sample_id'] for sample in test_dataset],
        },
        output_dir / 'dataset_split.json',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ppo.per_device_prompt_batch_size,
        shuffle=True,
        collate_fn=collator,  # 批处理函数，用于将单个样本列表合并成一个批次
        num_workers=config.data.num_workers,  # 用于数据加载的子进程数量
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.ppo.per_device_prompt_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if accelerator.is_main_process:
        print('Loading policy and reference models...')

    # 获得策略网络 Actor
    policy = build_policy_model(config.model, config.lora)

    # Reference Model（参考模型） 是一个参数被冻结的、与策略模型结构相同的模型，
    # 主要用于计算 KL 散度惩罚项，防止策略模型在优化过程中过度偏离原始行为（例如语言生成风格或事实性）
    reference_model = build_reference_model(config.model)

    optimizer = AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_epsilon,
        weight_decay=config.optimizer.weight_decay,
    )

    policy, reference_model, optimizer, train_loader, eval_loader = accelerator.prepare(
        policy,
        reference_model,
        optimizer,
        train_loader,
        eval_loader,
    )
    reference_model.eval()
    total_steps = len(train_loader) * config.num_train_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    if accelerator.is_main_process:
        print(
            '[ppo/setup] '
            f'train_samples={len(train_dataset)} '
            f'eval_samples={len(eval_dataset)} '
            f'prompt_batch_size={config.ppo.per_device_prompt_batch_size} '
            f'minibatch_size={config.ppo.per_device_minibatch_size} '
            f'ppo_epochs={config.ppo.ppo_epochs} '
            f'total_steps={total_steps}',
            flush=True,
        )

    if args.eval_only:
        metrics = run_evaluation(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            eval_loader=eval_loader,
            config=config,
            accelerator=accelerator,
            max_batches=args.max_steps,
        )
        if accelerator.is_main_process:
            append_metric(
                output_dir,
                {
                    'phase': 'eval',
                    'step': 0,
                    'epoch': -1,
                    'total_steps': total_steps,
                    **metrics,
                },
            )
            print('Eval metrics:', metrics)
            render_training_curve(output_dir)
        return

    global_step = 0
    for epoch in range(config.num_train_epochs):
        policy.train()
        for batch in train_loader:
            rollout = generate_rollout_batch(
                policy=policy,
                reference_model=reference_model,
                processor=processor,
                batch=batch,
                generation_config=config.generation,
                ppo_config=config.ppo,
                accelerator=accelerator,
            )

            batch_size = rollout.sequences.shape[0]
            epoch_metrics: dict[str, float] = {}
            for _ in range(config.ppo.ppo_epochs):
                permutation = torch.randperm(batch_size, device=accelerator.device)
                for start in range(0, batch_size, config.ppo.per_device_minibatch_size):
                    indices = permutation[start : start + config.ppo.per_device_minibatch_size]
                    minibatch = build_minibatch(
                        rollout=rollout,
                        indices=indices,
                        whiten_advantages=config.ppo.whiten_advantages,
                        device=accelerator.device,
                    )
                    loss_dict = compute_ppo_losses(
                        policy=policy,
                        minibatch=minibatch,
                        cliprange=config.ppo.cliprange,
                        value_cliprange=config.ppo.value_cliprange,
                        vf_coef=config.ppo.vf_coef,
                        entropy_coef=config.ppo.entropy_coef,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss_dict['loss'])
                    accelerator.clip_grad_norm_(policy.parameters(), config.optimizer.max_grad_norm)
                    optimizer.step()
                    epoch_metrics = {key: float(value.detach().float().item()) for key, value in loss_dict.items()}

            global_step += 1
            train_metrics = summarize_rollout(rollout)
            train_metrics.update(epoch_metrics)
            train_metrics['epoch'] = float(epoch)
            train_metrics['step'] = float(global_step)
            train_metrics['total_steps'] = float(total_steps)
            if accelerator.is_main_process:
                append_metric(output_dir, {'phase': 'train', **train_metrics})
            log_metrics(
                accelerator,
                prefix='train',
                metrics=train_metrics,
                total_steps=total_steps,
            )

            if global_step % config.logging.eval_steps == 0:
                eval_metrics = run_evaluation(
                    policy=policy,
                    reference_model=reference_model,
                    processor=processor,
                    eval_loader=eval_loader,
                    config=config,
                    accelerator=accelerator,
                )
                eval_metrics['step'] = float(global_step)
                eval_metrics['epoch'] = float(epoch)
                eval_metrics['total_steps'] = float(total_steps)
                if accelerator.is_main_process:
                    append_metric(output_dir, {'phase': 'eval', **eval_metrics})
                log_metrics(
                    accelerator,
                    prefix='eval',
                    metrics=eval_metrics,
                    total_steps=total_steps,
                )

            if global_step % config.logging.save_steps == 0 and accelerator.is_main_process:
                save_checkpoint(policy, optimizer, output_dir, global_step, config)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_eval = run_evaluation(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            eval_loader=eval_loader,
            config=config,
            accelerator=accelerator,
            max_batches=8,
        )
        final_eval['step'] = float(global_step)
        final_eval['epoch'] = float(config.num_train_epochs - 1)
        final_eval['total_steps'] = float(total_steps)
        append_metric(output_dir, {'phase': 'eval', **final_eval})
        log_metrics(
            accelerator,
            prefix='final_eval',
            metrics=final_eval,
            total_steps=total_steps,
        )
        save_checkpoint(policy, optimizer, output_dir, global_step, config)
        dump_json(
            {
                'test_size': len(test_dataset),
                'global_step': global_step,
                'total_steps': total_steps,
                'final_eval': final_eval,
            },
            output_dir / 'train_summary.json',
        )
        render_training_curve(output_dir)


if __name__ == '__main__':
    main()
