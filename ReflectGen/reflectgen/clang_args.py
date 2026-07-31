from __future__ import annotations


def normalize_standard(std: str | None) -> str | None:
    if not std:
        return None
    return std if std.startswith("-std=") else f"-std={std}"

