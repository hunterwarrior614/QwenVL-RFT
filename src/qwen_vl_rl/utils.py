from __future__ import annotations

import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import base64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def resolve_project_path(path: str | Path | None, project_root: str | Path) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.is_absolute():
        return target
    return Path(project_root) / target


def resolve_config_paths_in_dict(
    config: dict[str, Any],
    project_root: str | Path,
    required_keys: list[str],
    optional_keys: list[str] | None = None,
) -> dict[str, Any]:
    for key in required_keys:
        config[key] = str(resolve_project_path(config[key], project_root))

    for key in optional_keys or []:
        config[key] = (
            str(resolve_project_path(config[key], project_root))
            if config.get(key)
            else None
        )
    return config


def resolve_object_paths(
    obj: Any,
    project_root: str | Path,
    required_attrs: list[str],
    optional_attrs: list[str] | None = None,
) -> Any:
    for attr in required_attrs:
        setattr(obj, attr, str(resolve_project_path(getattr(obj, attr), project_root)))

    for attr in optional_attrs or []:
        value = getattr(obj, attr)
        setattr(
            obj,
            attr,
            str(resolve_project_path(value, project_root)) if value else None,
        )
    return obj


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def extract_first_image_uri(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for item in message.get('content', []):
            if item.get('type') == 'image':
                return item.get('image', '')
    return ''


def decode_data_uri_image(image_uri: str) -> Image.Image:
    encoded = image_uri.split(',', 1)[1] if image_uri.startswith('data:image') else image_uri
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes))
    return image.convert('RGB')


def resize_image_longest_edge(
    image: Image.Image,
    image_max_longest_edge: int | None = None,
) -> Image.Image:
    if image_max_longest_edge is None:
        return image

    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= image_max_longest_edge:
        return image

    scale = image_max_longest_edge / float(longest_edge)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    # 使用 Lanczos 算法将 image 缩放到 new_size
    return image.resize(new_size, Image.Resampling.LANCZOS)


def decode_first_image_from_messages(
    messages: list[dict[str, Any]],
    image_max_longest_edge: int | None = None,
) -> Image.Image:
    image_uri = extract_first_image_uri(messages)
    if not image_uri:
        raise ValueError('No image found in sample messages')

    return resize_image_longest_edge(
        decode_data_uri_image(image_uri),
        image_max_longest_edge=image_max_longest_edge,
    )
