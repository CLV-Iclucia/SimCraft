from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    column: int


@dataclass
class DirectBaseIR:
    qualified_name: str
    access: str = "public"
    is_virtual: bool = False
    location: SourceLocation | None = None


@dataclass
class ClassIR:
    kind: str
    name: str
    qualified_name: str
    namespace: str
    location: SourceLocation
    annotations: dict[str, str] = field(default_factory=dict)
    direct_base: DirectBaseIR | None = None
    has_rtti_body: bool = False
    type_symbol: str = ""
    depth: int = 0


@dataclass
class UnitIR:
    target: str
    source_target: str | None
    classes: list[ClassIR] = field(default_factory=list)


@dataclass
class RegistryIR:
    target: str
    units: list[str] = field(default_factory=list)
    classes: list[ClassIR] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)


def _to_dict(obj: Any) -> Any:
    return asdict(obj)


def _location_from_dict(data: dict[str, Any] | None) -> SourceLocation | None:
    if data is None:
        return None
    return SourceLocation(**data)


def _base_from_dict(data: dict[str, Any] | None) -> DirectBaseIR | None:
    if data is None:
        return None
    return DirectBaseIR(
        qualified_name=data["qualified_name"],
        access=data.get("access", "public"),
        is_virtual=data.get("is_virtual", False),
        location=_location_from_dict(data.get("location")),
    )


def _class_from_dict(data: dict[str, Any]) -> ClassIR:
    return ClassIR(
        kind=data.get("kind", "class"),
        name=data["name"],
        qualified_name=data["qualified_name"],
        namespace=data.get("namespace", ""),
        location=_location_from_dict(data["location"]),  # type: ignore[arg-type]
        annotations=data.get("annotations", {}),
        direct_base=_base_from_dict(data.get("direct_base")),
        has_rtti_body=data.get("has_rtti_body", False),
        type_symbol=data.get("type_symbol", ""),
        depth=data.get("depth", 0),
    )


def write_unit(unit: UnitIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(path, json.dumps(_to_dict(unit), indent=2))


def read_unit(path: Path) -> UnitIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return UnitIR(
        target=data["target"],
        source_target=data.get("source_target"),
        classes=[_class_from_dict(item) for item in data.get("classes", [])],
    )


def write_registry(registry: RegistryIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(path, json.dumps(_to_dict(registry), indent=2))


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")
