from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import torch

from .reward import extract_choice_letter


def extract_first_image_uri(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for item in message.get('content', []):
            if item.get('type') == 'image':
                return item.get('image', '')
    return ''


@torch.no_grad()
def generate_prediction_records(
    policy,
    processor,
    loader,
    accelerator,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model = accelerator.unwrap_model(policy)
    was_training = model.training
    model.eval()

    records: list[dict[str, Any]] = []
    for batch in loader:
        prompt_inputs = _move_tensors_to_device(batch['prompt_inputs'], accelerator.device)
        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
        sequences = generated.sequences if hasattr(generated, 'sequences') else generated
        prompt_length = prompt_inputs['input_ids'].shape[1]

        for row_idx, sample_id in enumerate(batch['sample_ids']):
            prediction = processor.tokenizer.decode(
                sequences[row_idx, prompt_length:],
                skip_special_tokens=True,
            ).strip()
            answer_key, ground_truth = _extract_target(batch, row_idx)
            pred_letter = extract_choice_letter(prediction)
            records.append(
                {
                    'sample_id': int(sample_id),
                    'prompt': _get_optional_list_value(batch, 'prompt_texts', row_idx, ''),
                    'question': _get_optional_list_value(batch, 'questions', row_idx, ''),
                    'answer_key': answer_key,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'pred_letter': pred_letter,
                    'correct': pred_letter == answer_key,
                    'image': extract_first_image_uri(batch['messages'][row_idx]),
                }
            )

    if was_training:
        model.train()
    return records


def write_prediction_report_from_loader(
    policy,
    processor,
    loader,
    accelerator,
    max_new_tokens: int,
    output_dir: str | Path,
    name: str = 'final_test_predictions',
) -> dict[str, str]:
    records = generate_prediction_records(
        policy=policy,
        processor=processor,
        loader=loader,
        accelerator=accelerator,
        max_new_tokens=max_new_tokens,
    )
    return write_prediction_report(records, output_dir=output_dir, name=name)


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
    {_render_optional_prompt(record)}
    <div class="label">Model Output</div>
    <pre>{html.escape(prediction)}</pre>
    <div class="label">Ground Truth</div>
    <pre>{html.escape(record.get('ground_truth', ''))}</pre>
  </div>
</article>'''


def _move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _extract_target(batch: dict[str, Any], row_idx: int) -> tuple[str | None, str]:
    if 'target_texts' in batch:
        ground_truth = batch['target_texts'][row_idx]
        return extract_choice_letter(ground_truth), ground_truth

    answer_key = _get_optional_list_value(batch, 'answer_keys', row_idx, None)
    ground_truth = _get_optional_list_value(batch, 'ground_truths', row_idx, answer_key or '')
    return answer_key, ground_truth


def _get_optional_list_value(
    batch: dict[str, Any],
    key: str,
    row_idx: int,
    default: Any,
) -> Any:
    values = batch.get(key)
    if values is None or row_idx >= len(values):
        return default
    return values[row_idx]


def _render_optional_prompt(record: dict[str, Any]) -> str:
    prompt = record.get('prompt', '')
    if not prompt:
        return ''
    return f'''<details>
      <summary class="label">Prompt</summary>
      <pre>{html.escape(prompt)}</pre>
    </details>'''
