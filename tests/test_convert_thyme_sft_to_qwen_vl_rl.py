from scripts.data_process.convert_thyme_sft_to_qwen_vl_rl import (
    ANSWER_FORMAT_PROMPT,
    build_common_fields,
    build_prompt_question,
)


def test_build_prompt_question_appends_answer_format():
    question = build_prompt_question('What is shown?')

    assert question.startswith('What is shown?')
    assert ANSWER_FORMAT_PROMPT.strip() in question
    assert '<answer>' in question
    assert '</answer>' in question
    assert "Your final answer to the user's question goes here." in question


def test_build_common_fields_embeds_format_instruction():
    record = build_common_fields(
        row_id=1,
        raw_image=None,
        raw_question='What is shown?',
        raw_response='<answer>B</answer>',
        image_mode='none',
        image_format='data_uri',
        keep_raw_fields=True,
        source_file='sample.parquet',
    )

    assert record['question'].startswith('What is shown?')
    assert ANSWER_FORMAT_PROMPT.strip() in record['question']
    assert record['base_question'] == 'What is shown?'
    assert record['messages'][0]['content'][0]['text'] == record['question']
    assert record['choice_letter'] == 'B'
    assert record['raw_question'] == 'What is shown?'
    assert record['raw_response'] == '<answer>B</answer>'


def test_build_common_fields_accepts_plain_choice_letter_source_response():
    record = build_common_fields(
        row_id=2,
        raw_image=None,
        raw_question='Which option is correct?',
        raw_response='c',
        image_mode='none',
        image_format='data_uri',
        keep_raw_fields=False,
        source_file='sample.parquet',
    )

    assert record['ground_truth'] == 'c'
    assert record['answer_normalized'] == 'C'
    assert record['choice_letter'] == 'C'
