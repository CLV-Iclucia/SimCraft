from __future__ import annotations

import re
from pathlib import Path

from .ir import ClassIR, DirectBaseIR, SourceLocation, UnitIR


CLASS_RE = re.compile(
    r"\b(?P<kind>class|struct)\s+"
    r"(?P<attrs>(?:(?:RF_CLASS|KLASS)\s*\([^)]*\)\s*)*)"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:final\s*)?"
    r"(?P<bases>:[^{;]+)?\{",
    re.MULTILINE,
)
NAMESPACE_RE = re.compile(r"\bnamespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\{")
RTTI_BODY_RE = re.compile(r"\b(?:RF_RTTI_BODY|KLASS_GENERATED_BODY)\s*\(")


def scan_files(
    input_files: list[Path],
    target: str,
    source_target: str | None = None,
) -> UnitIR:
    classes: list[ClassIR] = []
    for file_path in input_files:
        if not file_path.exists() or not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        classes.extend(_scan_text(text, file_path))
    return UnitIR(target=target, source_target=source_target, classes=classes)


def _scan_text(text: str, file_path: Path) -> list[ClassIR]:
    clean = _strip_comments(text)
    namespaces = _namespace_ranges(clean)
    classes: list[ClassIR] = []

    for match in CLASS_RE.finditer(clean):
        if not match.group("attrs"):
            continue

        body_start = match.end() - 1
        body_end = _find_matching_brace(clean, body_start)
        if body_end is None:
            continue

        name = match.group("name")
        namespace = _namespace_at(namespaces, match.start())
        qualified_name = f"{namespace}::{name}" if namespace else name
        line, column = _line_col(clean, match.start("name"))
        body = clean[body_start + 1 : body_end]

        classes.append(
            ClassIR(
                kind=match.group("kind"),
                name=name,
                qualified_name=qualified_name,
                namespace=namespace,
                location=SourceLocation(str(file_path), line, column),
                annotations={"class": "rf.class"},
                direct_base=_parse_direct_base(
                    match.group("bases"),
                    namespace,
                    clean,
                    file_path,
                    match.start(),
                    match.group("kind"),
                ),
                has_rtti_body=bool(RTTI_BODY_RE.search(body)),
            )
        )

    return classes


def _strip_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    depth = 0
    for i in range(open_pos, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _namespace_ranges(text: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for match in NAMESPACE_RE.finditer(text):
        open_pos = match.end() - 1
        close_pos = _find_matching_brace(text, open_pos)
        if close_pos is not None:
            ranges.append((match.start(), close_pos, match.group(1)))
    ranges.sort(key=lambda item: (item[0], item[1] - item[0]))
    return ranges


def _namespace_at(ranges: list[tuple[int, int, str]], pos: int) -> str:
    active = [name for start, end, name in ranges if start <= pos <= end]
    if not active:
        return ""
    parts: list[str] = []
    for name in active:
        parts.extend(name.split("::"))
    return "::".join(parts)


def _parse_direct_base(
    bases: str | None,
    namespace: str,
    text: str,
    file_path: Path,
    class_pos: int,
    kind: str,
) -> DirectBaseIR | None:
    if not bases:
        return None

    base_text = bases[1:].strip()
    chunks = [_normalize_base_chunk(chunk) for chunk in base_text.split(",")]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return None

    first = chunks[0]
    access = "public" if kind == "struct" else "private"
    is_virtual = False
    tokens = first.split()
    filtered: list[str] = []
    for token in tokens:
        if token in {"public", "protected", "private"}:
            access = token
        elif token == "virtual":
            is_virtual = True
        else:
            filtered.append(token)

    base_name = " ".join(filtered).strip()
    if base_name.startswith("::"):
        base_name = base_name[2:]

    line, column = _line_col(text, class_pos)
    return DirectBaseIR(
        qualified_name=base_name,
        access=access,
        is_virtual=is_virtual,
        location=SourceLocation(str(file_path), line, column),
    )


def _normalize_base_chunk(chunk: str) -> str:
    return " ".join(chunk.strip().split())


def _line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    last_newline = text.rfind("\n", 0, pos)
    column = pos + 1 if last_newline < 0 else pos - last_newline
    return line, column
