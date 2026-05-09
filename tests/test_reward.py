from src.qwen_vl_rl.reward import extract_choice_letter, score_choice_predictions


def test_extract_choice_letter_accepts_explicit_answer_formats():
    cases = {
        '<answer>a</answer>': 'A',
        '<think>reasoning with B inside</think><answer>C</answer>': 'C',
        '<answer>D.</answer>': 'D',
        '<answer> b: </answer>': 'B',
    }

    for text, expected in cases.items():
        assert extract_choice_letter(text) == expected


def test_extract_choice_letter_rejects_text_without_answer_tag():
    cases = [
        'A',
        ' b ',
        'C.',
        'D) because the image shows it',
        'Answer: d',
        'The correct answer is B.',
        '答案：C',
        'This is a small object.',
        'I can see a cat in the image.',
        'The answer is uncertain.',
        'Maybe the second option is correct.',
        '<think>A seems plausible</think>I am not sure.',
    ]

    for text in cases:
        assert extract_choice_letter(text) is None


def test_score_choice_predictions_uses_strict_extraction():
    output = score_choice_predictions(
        ['<answer>A</answer>', 'This is a small object.', '<answer>C</answer>'],
        ['A', 'A', 'B'],
    )

    assert output['pred_letters'] == ['A', None, 'C']
    assert output['rewards'] == [1.0, 0.0, 0.0]


def test_score_choice_predictions_rejects_plain_choice_letters_without_tags():
    output = score_choice_predictions(
        ['A', 'B', 'C'],
        ['A', 'B', 'D'],
    )

    assert output['pred_letters'] == [None, None, None]
    assert output['rewards'] == [0.0, 0.0, 0.0]
