from src.qwen_vl_rl.reward import (
    extract_choice_letter,
    extract_relaxed_choice_letter,
    score_choice_predictions,
    score_single_prediction,
)


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


def test_extract_relaxed_choice_letter_accepts_short_non_strict_outputs():
    cases = {
        'A': 'A',
        ' b ': 'B',
        'C.': 'C',
        'Answer: d': 'D',
        'The correct answer is probably B because the graph decreases.': 'B',
        '<answer>Option B</answer>': 'B',
    }

    for text, expected in cases.items():
        assert extract_relaxed_choice_letter(text) == expected


def test_extract_relaxed_choice_letter_rejects_ambiguous_or_long_outputs():
    cases = [
        'A or B',
        'This is a small object.',
        '<answer>A or C</answer>',
    ]

    for text in cases:
        assert extract_relaxed_choice_letter(text) is None


def test_score_choice_predictions_rewards_correctness_and_penalizes_wrong_options():
    output = score_choice_predictions(
        [
            '<answer>A</answer>',
            '<answer>Option A</answer>',
            'A',
            '<answer>C</answer>',
            'B',
            'This is a small object.',
        ],
        ['A', 'A', 'A', 'B', 'A', 'A'],
    )

    assert output['pred_letters'] == ['A', 'A', 'A', 'C', 'B', None]
    assert output['rewards'] == [1.0, 1.0, 1.0, -0.25, -0.25, -0.5]


def test_score_single_prediction_penalizes_unparseable_answer_tag():
    reward, pred_letter = score_single_prediction('<answer>uncertain</answer>', 'A')

    assert pred_letter is None
    assert reward == -0.5
