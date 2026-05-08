import json

from src.qwen_vl_rl.reports import extract_first_image_uri, write_prediction_report


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
