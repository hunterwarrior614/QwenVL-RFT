from types import SimpleNamespace

from src.qwen_vl_rl.collator_utils import (
    build_processor_inputs,
    build_generation_prompt_texts,
    collect_prompt_metadata,
    decode_prompt_images,
    prepare_tokenizer_for_padding,
)


class FakeTokenizer:
    def __init__(self):
        self.padding_side = 'right'
        self.pad_token = None
        self.eos_token = '<eos>'
        self.pad_token_id = None

    def convert_tokens_to_ids(self, token):
        if token == '<eos>':
            return 99
        raise AssertionError(f'unexpected token: {token}')


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        text_items = []
        for message in messages:
            for item in message.get('content', []):
                if item.get('type') == 'text':
                    text_items.append(item['text'])
        suffix = 'GEN' if add_generation_prompt else 'FULL'
        return f'{suffix}: ' + ' | '.join(text_items)

    def __call__(self, text, images, padding, return_tensors):
        return {
            'text': text,
            'images': images,
            'padding': padding,
            'return_tensors': return_tensors,
        }


def test_prepare_tokenizer_for_padding_sets_padding_side_and_pad_token():
    processor = FakeProcessor()

    prepare_tokenizer_for_padding(processor, padding_side='left')

    assert processor.tokenizer.padding_side == 'left'
    assert processor.tokenizer.pad_token == '<eos>'
    assert processor.tokenizer.pad_token_id == 99


def test_build_generation_prompt_texts_uses_chat_template():
    processor = FakeProcessor()
    batch = [
        {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Question A'},
                    ],
                }
            ]
        },
        {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Question B'},
                    ],
                }
            ]
        },
    ]

    prompt_texts = build_generation_prompt_texts(processor, batch)

    assert prompt_texts == ['GEN: Question A', 'GEN: Question B']


def test_decode_prompt_images_returns_one_image_per_sample():
    batch = [
        {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+y3YsAAAAASUVORK5CYII=',
                        },
                        {'type': 'text', 'text': 'Question'},
                    ],
                }
            ]
        }
    ]

    images = decode_prompt_images(batch, image_max_longest_edge=None)

    assert len(images) == 1
    assert images[0].size == (1, 1)


def test_build_processor_inputs_forwards_expected_arguments():
    processor = FakeProcessor()

    inputs = build_processor_inputs(
        processor,
        texts=['GEN: Question'],
        images=['fake-image'],
    )

    assert inputs == {
        'text': ['GEN: Question'],
        'images': ['fake-image'],
        'padding': True,
        'return_tensors': 'pt',
    }


def test_collect_prompt_metadata_collects_shared_fields():
    batch = [
        {
            'sample_id': 1,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q1'}]}],
            'question': 'Question 1',
        },
        {
            'sample_id': 2,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q2'}]}],
            'question': 'Question 2',
        },
    ]

    metadata = collect_prompt_metadata(batch)

    assert metadata == {
        'sample_ids': [1, 2],
        'messages': [
            [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q1'}]}],
            [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q2'}]}],
        ],
        'questions': ['Question 1', 'Question 2'],
    }
