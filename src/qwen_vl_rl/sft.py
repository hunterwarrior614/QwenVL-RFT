from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .data import decode_data_uri_image, resize_image_longest_edge


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
        self.processor.tokenizer.padding_side = 'right'
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.tokenizer.pad_token
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = []
        prompt_images = []
        full_texts = []
        sample_ids = []

        for sample in batch:
            prompt_texts.append(
                self.processor.apply_chat_template(
                    sample['messages'],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
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
            prompt_images.append(
                _decode_first_image(
                    sample['messages'],
                    image_max_longest_edge=self.image_max_longest_edge,
                )
            )
            sample_ids.append(sample['sample_id'])

        model_inputs = self.processor(
            text=full_texts,
            images=prompt_images,
            padding=True,
            return_tensors='pt',
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=prompt_images,
            padding=True,
            return_tensors='pt',
        )

        labels = model_inputs['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row_index in range(labels.shape[0]):
            prompt_len = int(prompt_inputs['attention_mask'][row_index].sum().item())
            labels[row_index, :prompt_len] = -100

        model_inputs['labels'] = labels
        return {
            'sample_ids': sample_ids,
            'model_inputs': model_inputs,
            'prompt_inputs': prompt_inputs,
            'target_texts': [sample['target_text'] for sample in batch],
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
            target_text=record.choice_letter,
            question=record.question,
        )
        for record in records
    ]


def _decode_first_image(
    messages: list[dict[str, Any]],
    image_max_longest_edge: int | None = None,
):
    for message in messages:
        for item in message.get('content', []):
            if item.get('type') == 'image':
                return resize_image_longest_edge(
                    decode_data_uri_image(item['image']),
                    image_max_longest_edge=image_max_longest_edge,
                )
    raise ValueError('No image found in sample messages')
