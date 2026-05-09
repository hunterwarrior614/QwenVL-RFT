from __future__ import annotations

import re


ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
EXACT_CHOICE_PATTERN = re.compile(r'^\s*([A-Da-d])\s*(?:[\)\].:：、-]\s*)?$')


def extract_answer_tag_content(text: str) -> str | None:
    if not text:
        return None

    match = ANSWER_TAG_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_choice_letter(text: str, require_answer_tag: bool = False) -> str | None:
    if not text:
        return None

    candidate = text
    if require_answer_tag:
        candidate = extract_answer_tag_content(text)
        if candidate is None:
            return None

    choice_match = EXACT_CHOICE_PATTERN.fullmatch(candidate.strip())
    if choice_match:
        return choice_match.group(1).upper()
    return None


def format_choice_answer(letter: str, with_answer_tag: bool = True) -> str:
    normalized = (letter or '').strip().upper()
    if normalized not in {'A', 'B', 'C', 'D'}:
        raise ValueError(f'Unsupported choice letter: {letter}')
    if with_answer_tag:
        return f'<answer>{normalized}</answer>'
    return normalized
