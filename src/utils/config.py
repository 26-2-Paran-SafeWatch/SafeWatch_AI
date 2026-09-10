"""설정 로더.

configs/ 아래 YAML을 로드한다. `extends: <파일명>` 키가 있으면 해당 파일을
먼저 로드한 뒤 현재 파일 값으로 얕은 병합(override)한다. (dev.yaml, pi5.yaml이
default.yaml을 상속하는 구조 — README.md, AGENTS.md 참고)

코드에서는 항상 이 로더를 거쳐 얻은 설정을 사용하고, 임계값을 직접 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """dict를 감싸 `cfg.risk.min_indicators` 같은 dot-access를 제공한다."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"설정에 '{name}' 키가 없습니다. configs/*.yaml을 확인하세요."
            ) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    parent = raw.get("extends")
    if parent:
        parent_path = path.parent / parent
        base = load_config(parent_path).to_dict()
        merged = _deep_merge(base, raw)
    else:
        merged = raw

    return Config(merged)
