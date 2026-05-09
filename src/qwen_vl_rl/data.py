from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .collator_utils import (
    build_processor_inputs,
    build_generation_prompt_texts,
    collect_prompt_metadata,
    decode_prompt_images,
    prepare_tokenizer_for_padding,
)


@dataclass
class ThymeVLPPORecord:
    sample_id: int
    messages: list[dict[str, Any]]
    question: str
    choice_letter: str
    ground_truth: str
    reference_answer: str


@dataclass
class ThymeVLGRPORecord:
    sample_id: int
    prompt: list[dict[str, Any]]
    question: str
    choice_letter: str
    reward_target: str
    ground_truth: str


class ThymeVLPPOJsonlDataset(Dataset):
    def __init__(self, records: list[ThymeVLPPORecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            'sample_id': record.sample_id,
            'messages': copy.deepcopy(record.messages),
            'question': record.question,
            'choice_letter': record.choice_letter,
            'ground_truth': record.ground_truth,
            'reference_answer': record.reference_answer,
        }


class ThymeVLGRPOJsonlDataset(Dataset):
    def __init__(self, records: list[ThymeVLGRPORecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            'sample_id': record.sample_id,
            'prompt': copy.deepcopy(record.prompt),
            'question': record.question,
            'choice_letter': record.choice_letter,
            'reward_target': record.reward_target,
            'ground_truth': record.ground_truth,
        }


# 一个自定义的数据整理器（collate function），专门用于为 Qwen VL（视觉语言模型）准备 PPO 训练所需的批次数据。
# 主要作用是将原始样本列表（每个样本包含对话消息、图像信息等）整理成一个统一的字典，其中包含：
#       1. 模型可以直接使用的输入张量（通过 processor 处理）
#       2. 训练所需的元数据（如样本 ID、答案选项、原始问题）
class QwenVLPPOCollator:
    def __init__(self, processor, image_max_longest_edge: int | None = None):
        self.processor = processor
        self.image_max_longest_edge = image_max_longest_edge

        # 生成任务中，需要将不同长度的 prompt 在左侧补齐，这样可以保证生成时新 token 追加在右侧，且 attention mask 计算正确
        prepare_tokenizer_for_padding(self.processor, padding_side='left')

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = build_generation_prompt_texts(self.processor, batch)
        prompt_images = decode_prompt_images(
            batch,
            image_max_longest_edge=self.image_max_longest_edge,
        )
        metadata = collect_prompt_metadata(batch)
        answer_keys = []
        ground_truths = []

        for sample in batch:
            answer_keys.append(sample['choice_letter'])
            ground_truths.append(sample.get('ground_truth', sample['choice_letter']))

        inputs = build_processor_inputs(self.processor, prompt_texts, prompt_images)
        return {
            'sample_ids': metadata['sample_ids'],
            'answer_keys': answer_keys,
            'questions': metadata['questions'],
            'ground_truths': ground_truths,
            'messages': [copy.deepcopy(messages) for messages in metadata['messages']],
            'prompt_texts': prompt_texts,
            'prompt_images': prompt_images,
            'prompt_inputs': inputs,
        }
    """
    假设 batch size 为 2, 则经过 __call__ 整理的 batch 内容形如：
    {
        "sample_ids": [1001, 1002],
        "answer_keys": ["B", "A"],
        "questions": ["What is the color of the car?", "How many apples are there?"],
        "prompt_texts": [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision|>What is the color of the car?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision|>How many apples are there?<|im_end|>\n<|im_start|>assistant\n"
        ],
        "prompt_images": [<PIL.Image>, <PIL.Image>],
        "prompt_inputs": {
            "input_ids": torch.tensor([[151644, 151645, ..., 151649], [151644, 151645, ..., 151649]]),
            "attention_mask": torch.tensor([[1, 1, ..., 0], [1, 1, ..., 0]]),
            "pixel_values": torch.tensor([...]),  // shape: (total_patches, 3, 448, 448)
            "image_grid_thw": torch.tensor([[1, 28, 28], [1, 28, 28]])
        }
    }
    """


class QwenVLGRPOCollator(QwenVLPPOCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ppo_batch = [
            {
                'sample_id': sample['sample_id'],
                'messages': sample['prompt'],
                'question': sample['question'],
                'choice_letter': sample['reward_target'],
                'ground_truth': sample['ground_truth'],
            }
            for sample in batch
        ]
        output = super().__call__(ppo_batch)
        output['reward_targets'] = [sample['reward_target'] for sample in batch]
        return output


def load_ppo_records(jsonl_path: str | Path) -> list[ThymeVLPPORecord]:
    records: list[ThymeVLPPORecord] = []
    path = Path(jsonl_path)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            payload = json.loads(line)
            choice_letter = (payload.get('choice_letter') or '').strip().upper()
            if choice_letter not in {'A', 'B', 'C', 'D'}:
                continue
            records.append(
                ThymeVLPPORecord(
                    sample_id=int(payload['id']),
                    messages=payload['messages'],
                    question=payload.get('question', ''),
                    choice_letter=choice_letter,
                    ground_truth=payload.get('ground_truth', ''),
                    reference_answer=payload.get('reference_answer', ''),
                )
            )
    return records


def load_grpo_records(jsonl_path: str | Path) -> list[ThymeVLGRPORecord]:
    records: list[ThymeVLGRPORecord] = []
    path = Path(jsonl_path)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            payload = json.loads(line)
            reward_target = (
                payload.get('reward_target') or payload.get('choice_letter') or ''
            ).strip().upper()
            if reward_target not in {'A', 'B', 'C', 'D'}:
                continue
            records.append(
                ThymeVLGRPORecord(
                    sample_id=int(payload['id']),
                    prompt=payload['prompt'],
                    question=payload.get('question', ''),
                    choice_letter=(payload.get('choice_letter') or reward_target).strip().upper(),
                    reward_target=reward_target,
                    ground_truth=payload.get('ground_truth', ''),
                )
            )
    return records


def create_split_datasets(
    jsonl_path: str | Path,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[ThymeVLPPOJsonlDataset, ThymeVLPPOJsonlDataset, ThymeVLPPOJsonlDataset]:
    records = load_ppo_records(jsonl_path)
    total_needed = train_size + eval_size + test_size
    if len(records) < total_needed:
        raise ValueError(
            f'Not enough PPO records: need {total_needed}, found {len(records)} in {jsonl_path}'
        )

    rng = random.Random(split_seed)
    rng.shuffle(records)

    train_records = records[:train_size]
    eval_records = records[train_size : train_size + eval_size]
    test_records = records[train_size + eval_size : total_needed]

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_eval_samples is not None:
        eval_records = eval_records[:max_eval_samples]

    return (
        ThymeVLPPOJsonlDataset(train_records),
        ThymeVLPPOJsonlDataset(eval_records),
        ThymeVLPPOJsonlDataset(test_records),
    )


def create_grpo_split_datasets(
    jsonl_path: str | Path,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[ThymeVLGRPOJsonlDataset, ThymeVLGRPOJsonlDataset, ThymeVLGRPOJsonlDataset]:
    records = load_grpo_records(jsonl_path)
    total_needed = train_size + eval_size + test_size
    if len(records) < total_needed:
        raise ValueError(
            f'Not enough GRPO records: need {total_needed}, found {len(records)} in {jsonl_path}'
        )

    rng = random.Random(split_seed)
    rng.shuffle(records)

    train_records = records[:train_size]
    eval_records = records[train_size : train_size + eval_size]
    test_records = records[train_size + eval_size : total_needed]

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_eval_samples is not None:
        eval_records = eval_records[:max_eval_samples]

    return (
        ThymeVLGRPOJsonlDataset(train_records),
        ThymeVLGRPOJsonlDataset(eval_records),
        ThymeVLGRPOJsonlDataset(test_records),
    )
