from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .collator_utils import (
    build_processor_inputs,
    build_processor_inputs_with_padding_side,
    build_generation_prompt_texts,
    collect_prompt_metadata,
    decode_prompt_images,
    prepare_tokenizer_for_padding,
)


@dataclass
class SFTRecord:
    sample_id: int
    messages: list[dict[str, Any]]
    target_text: str
    question: str


class ThymeVLSFTDataset(Dataset):
    def __init__(self, records: list[SFTRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            'sample_id': record.sample_id,
            'messages': copy.deepcopy(record.messages),
            'target_text': record.target_text,
            'question': record.question,
        }


class QwenVLSFTCollator:
    def __init__(self, processor, image_max_longest_edge: int | None = None):
        self.processor = processor
        self.image_max_longest_edge = image_max_longest_edge
        prepare_tokenizer_for_padding(self.processor, padding_side='right')

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = build_generation_prompt_texts(self.processor, batch)
        prompt_images = decode_prompt_images(
            batch,
            image_max_longest_edge=self.image_max_longest_edge,
        )
        full_texts = []
        metadata = collect_prompt_metadata(batch)
        target_texts = []

        for sample in batch:
            full_messages = copy.deepcopy(sample['messages'])
            full_messages.append(
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'text',
                            'text': sample['target_text'],
                        }
                    ],
                }
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
        )
            target_texts.append(sample['target_text'])

        model_inputs = build_processor_inputs_with_padding_side(
            self.processor,
            full_texts,
            prompt_images,
            padding_side='right',
        )
        prompt_inputs = build_processor_inputs_with_padding_side(
            self.processor,
            prompt_texts,
            prompt_images,
            padding_side='left',
        )

        labels = model_inputs['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row_index in range(labels.shape[0]):
            prompt_len = int(prompt_inputs['attention_mask'][row_index].sum().item())
            labels[row_index, :prompt_len] = -100

        model_inputs['labels'] = labels
        return {
            'sample_ids': metadata['sample_ids'],
            'messages': [copy.deepcopy(messages) for messages in metadata['messages']],
            'questions': metadata['questions'],
            'prompt_texts': prompt_texts,
            'model_inputs': model_inputs,
            'prompt_inputs': prompt_inputs,
            'target_texts': target_texts,
        }


def create_sft_datasets_from_ppo_records(
    jsonl_path: str,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[ThymeVLSFTDataset, ThymeVLSFTDataset, ThymeVLSFTDataset]:
    from .data import create_split_datasets

    train_ppo, eval_ppo, test_ppo = create_split_datasets(
        jsonl_path=jsonl_path,
        train_size=train_size,
        eval_size=eval_size,
        test_size=test_size,
        split_seed=split_seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    return (
        ThymeVLSFTDataset(_convert_records(train_ppo.records)),
        ThymeVLSFTDataset(_convert_records(eval_ppo.records)),
        ThymeVLSFTDataset(_convert_records(test_ppo.records)),
    )


def _convert_records(records) -> list[SFTRecord]:
    return [
        SFTRecord(
            sample_id=record.sample_id,
            messages=record.messages,
            target_text=f'<answer>{record.choice_letter}</answer>',
            question=record.question,
        )
        for record in records
    ]
