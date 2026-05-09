#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Iterable

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen_vl_rl.answering import extract_choice_letter as extract_choice_letter_from_text

DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_INPUT_PATH = Path("../Thyme-SFT/data/wo_thinking_thyme_single_round-00000-of-00146.parquet")


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"</?[^>]+>")
QUESTION_CUTOFF_PATTERNS = (
    re.compile(r"\n###\s*User Image Path", re.IGNORECASE),
    re.compile(r"\n###\s*User Image Size", re.IGNORECASE),
    re.compile(r"\n###\s*\*\*Output Format", re.IGNORECASE),
    re.compile(r"\n###\s*Output Format", re.IGNORECASE),
)
ANSWER_FORMAT_PROMPT = (
    "\n\n### Output Format\n"
    "<answer>Your final answer to the user's question goes here.</answer>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Thyme SFT trajectory parquet into Qwen-2.5-VL-friendly RL datasets "
            "for PPO and GRPO."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated RL dataset files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output filename prefix. Defaults to the input stem.",
    )
    parser.add_argument(
        "--export-targets",
        choices=("ppo", "grpo", "both"),
        default="both",
        help="Which RL dataset versions to export.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("first", "all", "none"),
        default="first",
        help=(
            "How many source images to keep. For direct Qwen-VL VQA training, "
            "'first' is usually the right choice."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=("data_uri", "raw_base64"),
        default="data_uri",
        help="How to serialize images for Qwen-VL message content.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of rows to process per parquet batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for debugging.",
    )
    parser.add_argument(
        "--keep-raw-fields",
        action="store_true",
        help="Keep raw SFT question/response and extracted reasoning in outputs.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    stripped_lines = [line.rstrip() for line in text.strip().splitlines()]
    normalized = "\n".join(stripped_lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def clean_question(question: str | None) -> str:
    if not question:
        return ""

    text = question.replace("\r\n", "\n").strip()
    text = re.sub(r"^\s*<image>\s*", "", text, flags=re.IGNORECASE)

    cut_positions = []
    for pattern in QUESTION_CUTOFF_PATTERNS:
        match = pattern.search(text)
        if match:
            cut_positions.append(match.start())
    if cut_positions:
        text = text[: min(cut_positions)]

    text = re.sub(r"\n\s*<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(text)


def build_prompt_question(question: str) -> str:
    question = clean_question(question)
    if not question:
        return ANSWER_FORMAT_PROMPT.strip()
    return f"{question}{ANSWER_FORMAT_PROMPT}"


def extract_reasoning(response: str | None) -> str:
    if not response:
        return ""
    match = THINK_PATTERN.search(response)
    if not match:
        return ""
    return normalize_text(match.group(1))


def extract_answer(response: str | None) -> str:
    if not response:
        return ""

    match = ANSWER_PATTERN.search(response)
    if match:
        return normalize_text(match.group(1))

    text = THINK_PATTERN.sub("", response)
    text = TAG_PATTERN.sub("", text)
    return normalize_text(text)


def normalize_answer(answer: str) -> str:
    answer = normalize_text(answer)
    choice = extract_choice_letter(answer)
    if choice:
        return choice
    return answer


def extract_choice_letter(answer: str) -> str | None:
    normalized = normalize_text(answer or "")
    choice = extract_choice_letter_from_text(normalized, require_answer_tag=False)
    if choice:
        return choice

    compact = normalized.upper()
    if compact in {"A", "B", "C", "D"}:
        return compact
    return None


def select_images(raw_image: object, image_mode: str) -> list[str]:
    if image_mode == "none" or raw_image is None:
        return []

    if isinstance(raw_image, str):
        images = [raw_image]
    elif isinstance(raw_image, Iterable) and not isinstance(raw_image, (bytes, bytearray)):
        images = [item for item in raw_image if item]
    else:
        images = [str(raw_image)]

    if image_mode == "first":
        return images[:1]
    return images


def format_image(image_base64: str, image_format: str) -> str:
    if image_format == "raw_base64":
        return image_base64
    if image_base64.startswith("data:image"):
        return image_base64
    return f"data:image/jpeg;base64,{image_base64}"


def build_qwen_messages(question: str, images: list[str], image_format: str) -> list[dict]:
    content = []
    for image in images:
        content.append(
            {
                "type": "image",
                "image": format_image(image, image_format),
            }
        )
    content.append(
        {
            "type": "text",
            "text": question,
        }
    )
    return [{"role": "user", "content": content}]


def detect_task_type(question: str, choice_letter: str | None) -> str:
    if choice_letter or "The choices are listed below" in question:
        return "visual_multiple_choice_qa"
    return "visual_qa"


def build_common_fields(
    row_id: int,
    raw_image: object,
    raw_question: str | None,
    raw_response: str | None,
    image_mode: str,
    image_format: str,
    keep_raw_fields: bool,
    source_file: str,
) -> dict:
    source_images = select_images(raw_image, "all")
    kept_images = select_images(raw_image, image_mode)
    question = clean_question(raw_question)
    prompt_question = build_prompt_question(raw_question)
    answer = extract_answer(raw_response)
    answer_normalized = normalize_answer(answer)
    choice_letter = extract_choice_letter(answer)
    messages = build_qwen_messages(prompt_question, kept_images, image_format)

    record = {
        "id": row_id,
        "question": prompt_question,
        "base_question": question,
        "messages": messages,
        "ground_truth": answer,
        "answer": answer,
        "answer_normalized": answer_normalized,
        "choice_letter": choice_letter,
        "task_type": detect_task_type(question, choice_letter),
        "image_count": len(kept_images),
        "num_source_images": len(source_images),
        "source_file": source_file,
        "metadata": {
            "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "image_mode": image_mode,
            "image_format": image_format,
            "source_format": "thyme_sft_trajectory",
        },
    }

    if keep_raw_fields:
        record["raw_question"] = raw_question or ""
        record["raw_response"] = raw_response or ""
        record["reasoning"] = extract_reasoning(raw_response)

    return record


def build_ppo_record(common: dict) -> dict:
    record = {
        "id": common["id"],
        "messages": common["messages"],
        "question": common["question"],
        "base_question": common["base_question"],
        "ground_truth": common["ground_truth"],
        "reference_answer": common["answer"],
        "answer_normalized": common["answer_normalized"],
        "choice_letter": common["choice_letter"],
        "task_type": common["task_type"],
        "image_count": common["image_count"],
        "num_source_images": common["num_source_images"],
        "source_file": common["source_file"],
        "metadata": common["metadata"],
    }
    for key in ("raw_question", "raw_response", "reasoning"):
        if key in common:
            record[key] = common[key]
    return record


def build_grpo_record(common: dict) -> dict:
    reward_target = common["choice_letter"] or common["answer_normalized"]
    record = {
        "id": common["id"],
        "prompt": common["messages"],
        "question": common["question"],
        "base_question": common["base_question"],
        "ground_truth": common["ground_truth"],
        "answer_normalized": common["answer_normalized"],
        "choice_letter": common["choice_letter"],
        "reward_target": reward_target,
        "reward_type": "exact_match",
        "task_type": common["task_type"],
        "image_count": common["image_count"],
        "num_source_images": common["num_source_images"],
        "source_file": common["source_file"],
        "metadata": common["metadata"],
    }
    for key in ("raw_question", "raw_response", "reasoning"):
        if key in common:
            record[key] = common[key]
    return record


def iter_common_records(
    input_path: Path,
    batch_size: int,
    image_mode: str,
    image_format: str,
    keep_raw_fields: bool,
    limit: int | None,
):
    parquet_file = pq.ParquetFile(input_path)
    row_id = 0
    processed = 0

    for batch in parquet_file.iter_batches(
        columns=["image", "question", "response"],
        batch_size=batch_size,
    ):
        batch_dict = batch.to_pydict()
        batch_rows = len(batch_dict["question"])
        for index in range(batch_rows):
            if limit is not None and processed >= limit:
                return

            yield build_common_fields(
                row_id=row_id,
                raw_image=batch_dict["image"][index],
                raw_question=batch_dict["question"][index],
                raw_response=batch_dict["response"][index],
                image_mode=image_mode,
                image_format=image_format,
                keep_raw_fields=keep_raw_fields,
                source_file=input_path.name,
            )
            row_id += 1
            processed += 1


def write_jsonl(output_path: Path, records) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_records(
    output_path: Path,
    common_records: list[dict],
    target: str,
) -> int:
    if target == "ppo":
        records = (build_ppo_record(record) for record in common_records)
    else:
        records = (build_grpo_record(record) for record in common_records)
    return write_jsonl(output_path, records)


def write_manifest(
    manifest_path: Path,
    input_path: Path,
    prefix: str,
    export_targets: str,
    image_mode: str,
    image_format: str,
    counts: dict[str, int],
) -> None:
    manifest = {
        "input": str(input_path),
        "output_prefix": prefix,
        "export_targets": export_targets,
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "image_mode": image_mode,
        "image_format": image_format,
        "files": counts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_absolute():
        args.input = PROJECT_ROOT / args.input
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    prefix = args.output_prefix or args.input.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_records = list(
        iter_common_records(
            input_path=args.input,
            batch_size=args.batch_size,
            image_mode=args.image_mode,
            image_format=args.image_format,
            keep_raw_fields=args.keep_raw_fields,
            limit=args.limit,
        )
    )

    counts: dict[str, int] = {}

    if args.export_targets in {"ppo", "both"}:
        ppo_path = args.output_dir / f"{prefix}.qwen_vl_ppo.jsonl"
        counts[str(ppo_path)] = export_records(ppo_path, common_records, "ppo")

    if args.export_targets in {"grpo", "both"}:
        grpo_path = args.output_dir / f"{prefix}.qwen_vl_grpo.jsonl"
        counts[str(grpo_path)] = export_records(grpo_path, common_records, "grpo")

    manifest_path = args.output_dir / f"{prefix}.qwen_vl_rl_manifest.json"
    write_manifest(
        manifest_path=manifest_path,
        input_path=args.input,
        prefix=prefix,
        export_targets=args.export_targets,
        image_mode=args.image_mode,
        image_format=args.image_format,
        counts=counts,
    )

    print(f"Converted {len(common_records)} rows")
    print(f"Input: {args.input}")
    for path_str, count in counts.items():
        print(f"Wrote {count} rows -> {path_str}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
