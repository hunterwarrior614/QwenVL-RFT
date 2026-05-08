from __future__ import annotations

import re
from typing import Iterable

ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
EXACT_CHOICE_PATTERN = re.compile(r'^\s*([A-Da-d])\s*(?:[\)\].:：、-]\s*)?$')


def extract_choice_letter(text: str) -> str | None:
    if not text:
        return None

    match = ANSWER_TAG_PATTERN.search(text)
    if not match:
        return None

    answer_text = match.group(1).strip()
    choice_match = EXACT_CHOICE_PATTERN.fullmatch(answer_text)
    if choice_match:
        return choice_match.group(1).upper()
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
