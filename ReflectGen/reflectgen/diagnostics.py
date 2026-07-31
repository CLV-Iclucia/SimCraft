from __future__ import annotations

from .ir import SourceLocation


class ReflectionError(RuntimeError):
    pass


def format_location(location: SourceLocation | None) -> str:
    if not location:
        return "<unknown>"
    return f"{location.file}:{location.line}:{location.column}"

