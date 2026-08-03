from __future__ import annotations

from pathlib import Path

from .ir import ClassIR, RegistryIR, write_registry


def generate_registry(registry: RegistryIR, output_dir: Path) -> tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_path = output_dir / f"{registry.target}.registry.json"

    user_headers = _classes_by_source_header(registry.classes)
    expected_generated_files = set()
    for source_file in user_headers:
        expected_generated_files.add(_generated_header_name(source_file))
        expected_generated_files.add(_generated_source_name(source_file))

    _remove_legacy_registry_files(output_dir, registry.target)
    _remove_stale_generated_files(output_dir, expected_generated_files)
    generated_files: list[Path] = []
    for source_file, classes in user_headers.items():
        generated_header = output_dir / _generated_header_name(source_file)
        generated_source = output_dir / _generated_source_name(source_file)
        _write_text_if_changed(generated_header, _render_user_header(registry, classes))
        _write_text_if_changed(
            generated_source,
            _render_user_source(registry, source_file, classes),
        )
        generated_files.extend([generated_header, generated_source])
    write_registry(registry, ir_path)

    return generated_files, ir_path


def _classes_by_source_header(classes: list[ClassIR]) -> dict[str, list[ClassIR]]:
    headers: dict[str, list[ClassIR]] = {}
    for class_ir in classes:
        path = Path(class_ir.location.file)
        if path.suffix.lower() not in {".h", ".hh", ".hpp", ".hxx"}:
            continue
        headers.setdefault(path.as_posix(), []).append(class_ir)
    return headers


def _generated_header_name(source_file: str) -> str:
    return f"{Path(source_file).stem}.generated.h"


def _generated_source_name(source_file: str) -> str:
    return f"{Path(source_file).stem}.generated.cpp"


def _remove_stale_generated_files(output_dir: Path, expected_names: set[str]) -> None:
    for path in output_dir.glob("*.generated.*"):
        if path.name not in expected_names:
            path.unlink()


def _remove_legacy_registry_files(output_dir: Path, target: str) -> None:
    for suffix in (".gen.h", ".gen.cpp"):
        path = output_dir / f"{target}{suffix}"
        if path.exists():
            path.unlink()


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _render_user_header(registry: RegistryIR, classes: list[ClassIR]) -> str:
    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append('#include "ReflectionMacros.h"')
    lines.append("")
    lines.extend(_forward_declarations(classes))
    lines.append("")
    lines.append("namespace RF {")
    lines.append("")

    for class_ir in _type_infos_required_by(registry, classes):
        lines.append(f"extern const TypeInfo {class_ir.type_symbol};")

    lines.append("")
    lines.append("template<typename T>")
    lines.append("struct StaticType;")
    lines.append("")

    for class_ir in classes:
        parent_symbol = _parent_symbol(registry, class_ir)
        lines.append("template<>")
        lines.append(f"struct StaticType<{class_ir.qualified_name}> {{")
        lines.append(
            f"    static constexpr TypeId Id() noexcept {{ return &{class_ir.type_symbol}; }}"
        )
        lines.append(
            f"    static constexpr TypeId Parent() noexcept {{ return {parent_symbol}; }}"
        )
        lines.append("};")
        lines.append("")

    lines.append("} // namespace RF")
    lines.append("")
    return "\n".join(lines)


def _render_user_source(
    registry: RegistryIR,
    source_file: str,
    classes: list[ClassIR],
) -> str:
    lines: list[str] = []
    lines.append(f'#include "{Path(source_file).as_posix()}"')
    lines.append("")
    lines.append("namespace RF {")
    lines.append("")

    for class_ir in classes:
        parent_symbol = _parent_symbol(registry, class_ir)
        lines.append(f"const TypeInfo {class_ir.type_symbol} = {{")
        lines.append(f"    {parent_symbol},")
        lines.append(f"    {class_ir.depth},")
        lines.append(f'    "{class_ir.qualified_name}",')
        lines.append("};")

    lines.append("")
    lines.append("} // namespace RF")
    lines.append("")

    for class_ir in classes:
        if class_ir.has_rtti_body:
            lines.append(f"RF::TypeId {class_ir.qualified_name}::GetTypeId() const {{")
            lines.append(f"    return RF::TypeOf<{class_ir.qualified_name}>();")
            lines.append("}")
            lines.append("")

    return "\n".join(lines)


def _type_infos_required_by(
    registry: RegistryIR,
    classes: list[ClassIR],
) -> list[ClassIR]:
    by_name = {item.qualified_name: item for item in registry.classes}
    required: set[str] = set()

    def add_with_parents(class_ir: ClassIR) -> None:
        if class_ir.qualified_name in required:
            return
        required.add(class_ir.qualified_name)
        if class_ir.direct_base:
            parent = by_name.get(class_ir.direct_base.qualified_name)
            if parent:
                add_with_parents(parent)

    for class_ir in classes:
        add_with_parents(class_ir)

    return [class_ir for class_ir in registry.classes
            if class_ir.qualified_name in required]


def _forward_declarations(classes: list[ClassIR]) -> list[str]:
    namespaces: dict[str, list[str]] = {}
    globals_: list[str] = []
    for class_ir in classes:
        if class_ir.namespace:
            namespaces.setdefault(class_ir.namespace, []).append(class_ir.name)
        else:
            globals_.append(class_ir.name)

    lines: list[str] = []
    for name in sorted(globals_):
        kind = _class_kind(classes, name)
        lines.append(f"{kind} {name};")
    for namespace in sorted(namespaces):
        class_names = " ".join(
            f"{_class_kind(classes, f'{namespace}::{name}')} {name};"
            for name in sorted(set(namespaces[namespace]))
        )
        lines.append(f"namespace {namespace} {{ {class_names} }}")
    return lines


def _class_kind(classes: list[ClassIR], name_or_qualified_name: str) -> str:
    for class_ir in classes:
        if (class_ir.qualified_name == name_or_qualified_name or
                class_ir.name == name_or_qualified_name):
            return class_ir.kind
    return "class"


def _parent_symbol(registry: RegistryIR, class_ir: ClassIR) -> str:
    if not class_ir.direct_base:
        return "InvalidType"
    by_name = {item.qualified_name: item for item in registry.classes}
    parent = by_name[class_ir.direct_base.qualified_name]
    return f"&{parent.type_symbol}"
