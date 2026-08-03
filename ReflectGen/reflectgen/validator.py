from __future__ import annotations

from collections import defaultdict

from .diagnostics import ReflectionError, format_location
from .ir import ClassIR, RegistryIR, UnitIR


def merge_units(target: str, units: list[UnitIR]) -> RegistryIR:
    classes: list[ClassIR] = []
    seen_units: list[str] = []
    for unit in units:
        seen_units.append(unit.target)
        classes.extend(unit.classes)

    registry = RegistryIR(target=target, units=seen_units, classes=classes)
    _resolve_base_names(registry)
    validate_registry(registry)
    _assign_symbols_and_depths(registry)
    _sort_classes(registry)
    return registry


def _resolve_base_names(registry: RegistryIR) -> None:
    by_name = {class_ir.qualified_name: class_ir for class_ir in registry.classes}
    for class_ir in registry.classes:
        base = class_ir.direct_base
        if not base:
            continue
        resolved = _resolve_type_name(base.qualified_name, class_ir.namespace, by_name)
        if resolved:
            base.qualified_name = resolved


def _resolve_type_name(
    spelling: str,
    current_namespace: str,
    by_name: dict[str, ClassIR],
) -> str | None:
    spelling = spelling.strip()
    if not spelling:
        return None

    candidates = _qualified_name_candidates(spelling, current_namespace)
    for candidate in candidates:
        if candidate in by_name:
            return candidate

    suffix = f"::{spelling}"
    suffix_matches = [
        qualified for qualified in by_name
        if qualified.endswith(suffix) or qualified == spelling
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        matches = ", ".join(sorted(suffix_matches))
        raise ReflectionError(
            f"ambiguous reflected type name {spelling}; candidates: {matches}"
        )
    return None


def _qualified_name_candidates(spelling: str, current_namespace: str) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(spelling)
    namespace_parts = current_namespace.split("::") if current_namespace else []
    for count in range(len(namespace_parts), -1, -1):
        prefix = "::".join(namespace_parts[:count])
        add(f"{prefix}::{spelling}" if prefix else spelling)
    return candidates


def validate_registry(registry: RegistryIR) -> None:
    by_name: dict[str, ClassIR] = {}
    duplicates: dict[str, list[ClassIR]] = defaultdict(list)
    for class_ir in registry.classes:
        duplicates[class_ir.qualified_name].append(class_ir)
        if class_ir.direct_base and "," in class_ir.direct_base.qualified_name:
            raise ReflectionError(
                f"class {class_ir.qualified_name} appears to use multiple inheritance at "
                f"{format_location(class_ir.location)}"
            )

    for name, items in duplicates.items():
        if len(items) > 1:
            locations = ", ".join(format_location(item.location) for item in items)
            raise ReflectionError(f"duplicate reflected class {name}: {locations}")
        by_name[name] = items[0]

    symbols: dict[str, ClassIR] = {}
    for class_ir in registry.classes:
        symbol = type_symbol_for(class_ir.qualified_name)
        if symbol in symbols:
            other = symbols[symbol]
            raise ReflectionError(
                f"TypeInfo symbol collision: {class_ir.qualified_name} and "
                f"{other.qualified_name} both map to {symbol}"
            )
        symbols[symbol] = class_ir

    for class_ir in registry.classes:
        base = class_ir.direct_base
        if not base:
            continue
        if base.is_virtual:
            raise ReflectionError(
                f"virtual base is not supported for {class_ir.qualified_name} at "
                f"{format_location(base.location)}"
            )
        if base.qualified_name not in by_name:
            raise ReflectionError(
                f"{class_ir.qualified_name} inherits from {base.qualified_name}, "
                "but that base was not found in the merged reflection registry"
            )

    _detect_cycles(by_name)


def type_symbol_for(qualified_name: str) -> str:
    qualified_name = qualified_name.replace("::", "_")
    cleaned = []
    for ch in qualified_name:
        cleaned.append(ch if ch.isalnum() or ch == "_" else "_")
    return "Type_" + "".join(cleaned).strip("_")


def _detect_cycles(by_name: dict[str, ClassIR]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ReflectionError(f"inheritance cycle detected at {name}")
        visiting.add(name)
        cls = by_name[name]
        if cls.direct_base and cls.direct_base.qualified_name in by_name:
            visit(cls.direct_base.qualified_name)
        visiting.remove(name)
        visited.add(name)

    for name in by_name:
        visit(name)


def _assign_symbols_and_depths(registry: RegistryIR) -> None:
    by_name = {class_ir.qualified_name: class_ir for class_ir in registry.classes}
    roots: list[str] = []

    def depth_of(class_ir: ClassIR) -> int:
        if class_ir.depth:
            return class_ir.depth
        class_ir.type_symbol = type_symbol_for(class_ir.qualified_name)
        if not class_ir.direct_base:
            if class_ir.qualified_name not in roots:
                roots.append(class_ir.qualified_name)
            class_ir.depth = 0
            return 0
        parent = by_name[class_ir.direct_base.qualified_name]
        class_ir.depth = depth_of(parent) + 1
        return class_ir.depth

    for class_ir in registry.classes:
        depth_of(class_ir)

    registry.roots = sorted(roots)


def _sort_classes(registry: RegistryIR) -> None:
    registry.classes.sort(key=lambda item: (item.depth, item.qualified_name))
