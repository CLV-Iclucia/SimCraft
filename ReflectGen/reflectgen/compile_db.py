from __future__ import annotations

from pathlib import Path


def find_compile_commands(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        candidate = directory / "compile_commands.json"
        if candidate.exists():
            return candidate
    return None

