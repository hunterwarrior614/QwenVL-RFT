from __future__ import annotations

from typing import Any

from .utils import decode_first_image_from_messages


def prepare_tokenizer_for_padding(processor, padding_side: str) -> None:
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.pad_token
    )


def build_generation_prompt_texts(processor, batch: list[dict[str, Any]]) -> list[str]:
    return [
        processor.apply_chat_template(
            sample['messages'],
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in batch
    ]


def decode_prompt_images(
    batch: list[dict[str, Any]],
    image_max_longest_edge: int | None = None,
) -> list[Any]:
    return [
        decode_first_image_from_messages(
            sample['messages'],
            image_max_longest_edge=image_max_longest_edge,
        )
        for sample in batch
    ]


def build_processor_inputs(processor, texts: list[str], images: list[Any]):
    return build_processor_inputs_with_padding_side(
        processor,
        texts=texts,
        images=images,
        padding_side=processor.tokenizer.padding_side,
    )


def build_processor_inputs_with_padding_side(
    processor,
    texts: list[str],
    images: list[Any],
    padding_side: str,
):
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = padding_side
    try:
        return processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors='pt',
        )
    finally:
        processor.tokenizer.padding_side = original_padding_side


def collect_prompt_metadata(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        'sample_ids': [sample['sample_id'] for sample in batch],
        'messages': [sample['messages'] for sample in batch],
        'questions': [sample['question'] for sample in batch],
    }
