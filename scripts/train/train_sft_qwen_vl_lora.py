#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen_vl_rl.modeling_common import (
    build_quantization_config_from_fields,
    get_torch_dtype,
    resolve_lora_target_modules,
)
from src.qwen_vl_rl.reports import write_test_results_from_loader
from src.qwen_vl_rl.answering import extract_choice_letter
from src.qwen_vl_rl.sft import QwenVLSFTCollator, create_sft_datasets_from_ppo_records
from src.qwen_vl_rl.training_io import (
    estimate_total_training_steps,
    prepare_checkpoint_dir,
    save_optimizer_and_training_state,
)
from src.qwen_vl_rl.utils import (
    dump_json,
    ensure_dir,
    move_tensors_to_device,
    resolve_config_paths_in_dict,
    resolve_project_path,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SFT warm start for Qwen2.5-VL LoRA on Thyme VQA data.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sft_qwen_vl_lora.yaml',
    )
    parser.add_argument('--max-steps', type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = resolve_project_path(path, PROJECT_ROOT)
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    return resolve_config_paths_in_dict(
        config,
        PROJECT_ROOT,
        required_keys=['output_dir', 'base_model_name_or_path', 'train_file'],
    )


def build_quantization_config(config: dict) -> BitsAndBytesConfig | None:
    return build_quantization_config_from_fields(config)


def resolve_lora_targets(model, target_regex: str) -> list[str]:
    return resolve_lora_target_modules(model, target_regex)


@torch.no_grad()
def evaluate(
    policy,
    processor,
    valid_loader,
    accelerator,
    max_new_tokens: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    policy.eval()
    losses = []
    exact = 0
    total = 0
    for batch_index, batch in enumerate(valid_loader):
        model_inputs = move_tensors_to_device(batch['model_inputs'], accelerator.device)
        outputs = policy(**model_inputs)
        losses.append(float(outputs.loss.detach().float().item()))

        prompt_inputs = move_tensors_to_device(batch['prompt_inputs'], accelerator.device)
        generated = accelerator.unwrap_model(policy).generate(
            input_ids=prompt_inputs['input_ids'],
            attention_mask=prompt_inputs['attention_mask'],
            pixel_values=prompt_inputs['pixel_values'],
            image_grid_thw=prompt_inputs['image_grid_thw'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        for row_idx, target in enumerate(batch['target_texts']):
            prediction = processor.tokenizer.decode(
                generated[row_idx, prompt_length:], skip_special_tokens=True
            ).strip()
            pred_letter = extract_choice_letter(prediction, require_answer_tag=True)
            target_letter = extract_choice_letter(target, require_answer_tag=True)
            exact += int(pred_letter == target_letter)
            total += 1

        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    stats = torch.tensor(
        [sum(losses), float(len(losses)), float(exact), float(total)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    if accelerator.num_processes > 1:
        stats = accelerator.reduce(stats, reduction='sum')

    policy.train()
    return {
        'eval_loss': float(stats[0].item() / max(float(stats[1].item()), 1.0)),
        'eval_exact_match': float(stats[2].item() / max(float(stats[3].item()), 1.0)),
    }


def save_checkpoint(model, processor, optimizer, output_dir: Path, step: int) -> None:
    checkpoint_dir = prepare_checkpoint_dir(output_dir, step)
    model.save_pretrained(checkpoint_dir / 'adapter')
    processor.save_pretrained(checkpoint_dir / 'processor')
    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={'step': step},
    )


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
    train_losses = [item['loss'] for item in records if item['phase'] == 'train']
    eval_steps = [item['step'] for item in records if item['phase'] == 'eval']
    eval_losses = [item['eval_loss'] for item in records if item['phase'] == 'eval']
    eval_accs = [item['eval_exact_match'] for item in records if item['phase'] == 'eval']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, label='train_loss', color='#1f77b4', linewidth=2)
    if eval_steps:
        axes[0].plot(eval_steps, eval_losses, label='eval_loss', color='#d62728', linewidth=2, marker='o')
    axes[0].set_title('SFT Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if eval_steps:
        axes[1].plot(eval_steps, eval_accs, label='eval_exact_match', color='#2ca02c', linewidth=2, marker='o')
    axes[1].set_title('SFT Eval Exact Match')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    if eval_steps:
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / 'training_curve.png', dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['seed'])

    output_dir = ensure_dir(config['output_dir'])

    # 每经过 gradient_accumulation_steps 个 mini-batch 才执行一次参数更新，
    # 当显存不足以支持较大的批次时，通过梯度累积来模拟更大的有效批次。
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    if accelerator.is_main_process:
        dump_json(config, output_dir / 'resolved_config.json')
        metrics_path = output_dir / 'metrics.jsonl'
        metrics_path.write_text('', encoding='utf-8')
    accelerator.wait_for_everyone()
    # 加载模型
    processor = AutoProcessor.from_pretrained(config['base_model_name_or_path'])

    train_dataset, valid_dataset, test_dataset = create_sft_datasets_from_ppo_records(
        jsonl_path=config['train_file'],
        train_size=config['train_size'],
        eval_size=config['eval_size'],
        test_size=config['test_size'],
        split_seed=config['split_seed'],
        max_train_samples=config.get('max_train_samples'),
        max_eval_samples=config.get('max_eval_samples'),
    )
    dump_json(
        {
            'train_size': len(train_dataset),
            'eval_size': len(valid_dataset),
            'test_size': len(test_dataset),
            'split_seed': config['split_seed'],
            'train_ids': [sample['sample_id'] for sample in train_dataset],
            'eval_ids': [sample['sample_id'] for sample in valid_dataset],
            'test_ids': [sample['sample_id'] for sample in test_dataset],
        },
        output_dir / 'dataset_split.json',
    )

    quantization_config = build_quantization_config(config)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config['base_model_name_or_path'],
        dtype=get_torch_dtype(config['torch_dtype']),
        attn_implementation=config['attn_implementation'],
        quantization_config=quantization_config,
    )
    if config.get('load_in_4bit', False):
        model = prepare_model_for_kbit_training(model)

    target_modules = resolve_lora_targets(model, config['lora_target_modules_regex'])
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias=config['lora_bias'],
        target_modules=target_modules,
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, peft_config)
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()

    collator = QwenVLSFTCollator(
        processor,
        image_max_longest_edge=config.get('image_max_longest_edge'),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['per_device_eval_batch_size'],
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['per_device_eval_batch_size'],
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    total_steps = estimate_total_training_steps(
        num_batches=len(train_loader),
        num_train_epochs=config['num_train_epochs'],
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_steps=args.max_steps,
    )
    warmup_steps = max(1, int(total_steps * config['warmup_ratio']))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if accelerator.is_main_process:
        print(
            '[sft/setup] '
            f"train_samples={len(train_dataset)} "
            f"valid_samples={len(valid_dataset)} "
            f"per_device_train_batch_size={config['per_device_train_batch_size']} "
            f"grad_accum={config['gradient_accumulation_steps']} "
            f"total_steps={total_steps} "
            f"warmup_steps={warmup_steps} "
            f"num_processes={accelerator.num_processes} "
            f"process_index={accelerator.process_index}",
            flush=True,
        )

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader,
        scheduler,
    )

    global_step = 0
    for epoch in range(config['num_train_epochs']):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                model_inputs = move_tensors_to_device(batch['model_inputs'], accelerator.device)
                outputs = model(**model_inputs)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config['max_grad_norm']
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    current_lr = float(scheduler.get_last_lr()[0])
                    train_record = {
                        'phase': 'train',
                        'step': global_step,
                        'epoch': epoch,
                        'loss': float(loss.detach().float().item()),
                        'lr': current_lr,
                        'total_steps': total_steps,
                    }
                    append_metric(output_dir, train_record)
                    if global_step % config['logging_steps'] == 0 or global_step == 1 or global_step == total_steps:
                        print(
                            '[sft/train] '
                            f"step={global_step}/{total_steps} "
                            f"loss={train_record['loss']:.4f} "
                            f"lr={current_lr:.6e}",
                            flush=True,
                        )
                if global_step % config['eval_steps'] == 0:
                    metrics = evaluate(
                        model,
                        processor,
                        valid_loader,
                        accelerator,
                        max_new_tokens=config['max_new_tokens_eval'],
                        max_batches=8,
                    )
                    if accelerator.is_main_process:
                        append_metric(
                            output_dir,
                            {
                                'phase': 'eval',
                                'step': global_step,
                                'epoch': epoch,
                                'eval_loss': float(metrics['eval_loss']),
                                'eval_exact_match': float(metrics['eval_exact_match']),
                                'total_steps': total_steps,
                            },
                        )
                        print(
                            '[sft/eval] '
                            f"step={global_step}/{total_steps} "
                            + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()),
                            flush=True,
                        )
                if (
                    global_step % config['save_steps'] == 0
                    and accelerator.is_main_process
                ):
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        processor,
                        optimizer,
                        output_dir,
                        global_step,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    final_metrics = evaluate(
        model,
        processor,
        valid_loader,
        accelerator,
        max_new_tokens=config['max_new_tokens_eval'],
    )
    if accelerator.is_main_process:
        test_output = write_test_results_from_loader(
            policy=model,
            processor=processor,
            loader=test_loader,
            accelerator=accelerator,
            max_new_tokens=config['max_new_tokens_eval'],
            output_dir=output_dir,
        )
        append_metric(
            output_dir,
            {
                'phase': 'eval',
                'step': global_step,
                'epoch': config['num_train_epochs'] - 1,
                'eval_loss': float(final_metrics['eval_loss']),
                'eval_exact_match': float(final_metrics['eval_exact_match']),
                'total_steps': total_steps,
            },
        )
        print(
            '[sft/final_eval] '
            f"step={global_step}/{total_steps} "
            + ' '.join(f'{k}={v:.4f}' for k, v in final_metrics.items()),
            flush=True,
        )
        save_checkpoint(
            accelerator.unwrap_model(model),
            processor,
            optimizer,
            output_dir,
            global_step,
        )
        dump_json(
            {
                'test_size': len(test_dataset),
                'global_step': global_step,
                'total_steps': total_steps,
                'final_eval': final_metrics,
                'test_metrics': test_output['metrics'],
                'test_results': test_output['paths'],
            },
            output_dir / 'train_summary.json',
        )
        render_training_curve(output_dir)


if __name__ == '__main__':
    main()
