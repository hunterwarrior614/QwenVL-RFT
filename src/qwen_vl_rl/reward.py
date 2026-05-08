from __future__ import annotations

import re
from typing import Iterable

# re.IGNORECASE → 忽略大小写
# re.DOTALL → 使得 . 匹配包括换行符 \n 在内的任意字符，从而可以跨行匹配
CHOICE_PATTERN = re.compile(r'\b([A-D])\b', re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)


def extract_choice_letter(text: str) -> str | None:
    if not text:
        return None

    match = ANSWER_TAG_PATTERN.search(text)
    if match:
        text = match.group(1)

    text = text.strip()
    if text.upper() in {'A', 'B', 'C', 'D'}:
        return text.upper()

    match = CHOICE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


def score_choice_predictions(predictions: Iterable[str], answer_keys: Iterable[str]) -> dict[str, list[float] | list[str | None]]:
    rewards: list[float] = []
    extracted: list[str | None] = []
    for prediction, answer_key in zip(predictions, answer_keys):
        pred_letter = extract_choice_letter(prediction)
        extracted.append(pred_letter)
        rewards.append(1.0 if pred_letter == answer_key else 0.0)
    return {
        'rewards': rewards,
        'pred_letters': extracted,
    }
