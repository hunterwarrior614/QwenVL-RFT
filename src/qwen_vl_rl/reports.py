from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def extract_first_image_uri(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for item in message.get('content', []):
            if item.get('type') == 'image':
                return item.get('image', '')
    return ''


def write_prediction_report(records: list[dict[str, Any]], output_dir: str | Path, name: str) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    jsonl_path = output / f'{name}.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    html_path = output / f'{name}.html'
    html_path.write_text(render_prediction_report_html(records, title=name), encoding='utf-8')

    return {
        'jsonl': str(jsonl_path),
        'html': str(html_path),
    }


def render_prediction_report_html(records: list[dict[str, Any]], title: str) -> str:
    total = len(records)
    correct = sum(1 for record in records if record.get('correct'))
    accuracy = correct / max(total, 1)
    rows = '\n'.join(_render_record(record) for record in records)

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f6f8fa;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 16px 24px;
      border-bottom: 1px solid #d9e2ec;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 650;
    }}
    .summary {{
      display: flex;
      gap: 16px;
      font-size: 14px;
      color: #52606d;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 16px;
      padding: 16px;
    }}
    article {{
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #ffffff;
    }}
    img {{
      width: 190px;
      max-height: 260px;
      object-fit: contain;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      background: #f0f4f8;
    }}
    .content {{
      min-width: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 12px;
      color: #52606d;
    }}
    .badge {{
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f7;
    }}
    .correct {{
      color: #176f3d;
      background: #e3f9e5;
    }}
    .wrong {{
      color: #9b1c1c;
      background: #ffe3e3;
    }}
    pre {{
      margin: 6px 0 10px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #243b53;
    }}
    .label {{
      margin-top: 8px;
      font-size: 12px;
      font-weight: 650;
      color: #334e68;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="summary">
      <span>total={total}</span>
      <span>correct={correct}</span>
      <span>accuracy={accuracy:.4f}</span>
    </div>
  </header>
  <main>
    {rows}
  </main>
</body>
</html>
'''


def _render_record(record: dict[str, Any]) -> str:
    image_uri = record.get('image', '')
    status_class = 'correct' if record.get('correct') else 'wrong'
    status_text = 'correct' if record.get('correct') else 'wrong'
    prediction = record.get('prediction', '')
    pred_letter = record.get('pred_letter')
    answer_key = record.get('answer_key', '')

    image_html = (
        f'<img src="{html.escape(image_uri, quote=True)}" alt="sample image">'
        if image_uri
        else '<div class="image-missing">No image</div>'
    )
    return f'''<article>
  <div>{image_html}</div>
  <div class="content">
    <div class="meta">
      <span class="badge">id={html.escape(str(record.get('sample_id', '')))}</span>
      <span class="badge {status_class}">{status_text}</span>
      <span class="badge">target={html.escape(str(answer_key))}</span>
      <span class="badge">pred={html.escape(str(pred_letter))}</span>
    </div>
    <div class="label">Question</div>
    <pre>{html.escape(record.get('question', ''))}</pre>
    <div class="label">Model Output</div>
    <pre>{html.escape(prediction)}</pre>
    <div class="label">Ground Truth</div>
    <pre>{html.escape(record.get('ground_truth', ''))}</pre>
  </div>
</article>'''
