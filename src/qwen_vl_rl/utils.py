from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
