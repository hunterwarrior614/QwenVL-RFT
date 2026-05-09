from __future__ import annotations

from typing import Iterable

from .answering import extract_choice_letter as _extract_choice_letter


def extract_choice_letter(text: str) -> str | None:
    return _extract_choice_letter(text, require_answer_tag=True)


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
