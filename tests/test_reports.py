import json

import torch

from src.qwen_vl_rl.reports import (
    extract_first_image_uri,
    generate_prediction_records,
    write_prediction_report,
)


def test_extract_first_image_uri_returns_first_image():
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'question'},
                {'type': 'image', 'image': 'data:image/png;base64,abc'},
            ],
        }
    ]

    assert extract_first_image_uri(messages) == 'data:image/png;base64,abc'


def test_write_prediction_report_writes_jsonl_and_html(tmp_path):
    records = [
        {
            'sample_id': 1,
            'question': 'Q?',
            'answer_key': 'A',
            'ground_truth': '<answer>A</answer>',
            'raw_response': '  <answer>A</answer>\n',
            'prediction': '<answer>A</answer>',
            'pred_letter': 'A',
            'correct': True,
            'image': 'data:image/png;base64,abc',
        }
    ]

    paths = write_prediction_report(records, tmp_path, 'report')
    jsonl_path = tmp_path / 'report.jsonl'
    html_path = tmp_path / 'report.html'

    assert paths == {'jsonl': str(jsonl_path), 'html': str(html_path)}
    assert json.loads(jsonl_path.read_text(encoding='utf-8').strip()) == records[0]
    html = html_path.read_text(encoding='utf-8')
    assert 'Q?' in html
    assert '&lt;answer&gt;A&lt;/answer&gt;' in html
    assert 'data:image/png;base64,abc' in html


def test_generate_prediction_records_from_loader():
    class FakeAccelerator:
        device = torch.device('cpu')

        def unwrap_model(self, model):
            return model

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def decode(self, tokens, skip_special_tokens=True):
            return '  <answer>A</answer>\n'

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakePolicy(torch.nn.Module):
        def generate(self, **kwargs):
            input_ids = kwargs['input_ids']
            response = torch.tensor([[101, 102]], dtype=input_ids.dtype)
            return torch.cat([input_ids, response], dim=1)

    loader = [
        {
            'prompt_inputs': {
                'input_ids': torch.tensor([[11, 12]]),
                'attention_mask': torch.tensor([[1, 1]]),
            },
            'sample_ids': [7],
            'prompt_texts': ['prompt text'],
            'questions': ['Q?'],
            'answer_keys': ['A'],
            'ground_truths': ['A'],
            'messages': [[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': 'data:image/png;base64,abc'},
                    ],
                }
            ]],
        }
    ]

    policy = FakePolicy()
    policy.train()

    records = generate_prediction_records(
        policy=policy,
        processor=FakeProcessor(),
        loader=loader,
        accelerator=FakeAccelerator(),
        max_new_tokens=2,
    )

    assert policy.training
    assert records == [
        {
            'sample_id': 7,
            'prompt': 'prompt text',
            'question': 'Q?',
            'answer_key': 'A',
            'ground_truth': 'A',
            'raw_response': '  <answer>A</answer>\n',
            'prediction': '<answer>A</answer>',
            'pred_letter': 'A',
            'correct': True,
            'image': 'data:image/png;base64,abc',
        }
    ]
