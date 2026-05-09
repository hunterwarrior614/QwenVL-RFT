from src.qwen_vl_rl.answering import (
    extract_answer_tag_content,
    extract_choice_letter,
    format_choice_answer,
)


def test_extract_answer_tag_content_returns_inner_text():
    assert extract_answer_tag_content('<answer> B </answer>') == 'B'
    assert extract_answer_tag_content('<think>...</think><answer>C.</answer>') == 'C.'
    assert extract_answer_tag_content('A') is None


def test_extract_choice_letter_supports_strict_and_non_strict_modes():
    assert extract_choice_letter('<answer>A</answer>', require_answer_tag=True) == 'A'
    assert extract_choice_letter('B', require_answer_tag=True) is None

    assert extract_choice_letter('<answer>C</answer>', require_answer_tag=False) is None
    assert extract_choice_letter('D', require_answer_tag=False) == 'D'
    assert extract_choice_letter(' b: ', require_answer_tag=False) == 'B'


def test_format_choice_answer_formats_and_validates_letters():
    assert format_choice_answer('a') == '<answer>A</answer>'
    assert format_choice_answer('b', with_answer_tag=False) == 'B'
