from __future__ import annotations

from pathlib import Path

from .ir import ClassIR, RegistryIR, write_registry


def generate_registry(registry: RegistryIR, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / f"{registry.target}.gen.h"
    source_path = output_dir / f"{registry.target}.gen.cpp"
    ir_path = output_dir / f"{registry.target}.registry.json"

    header_path.write_text(_render_header(registry), encoding="utf-8")
    source_path.write_text(_render_source(registry, header_path.name), encoding="utf-8")
    write_registry(registry, ir_path)

    return header_path, source_path, ir_path


def _render_header(registry: RegistryIR) -> str:
    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append('#include "ReflectionMacros.h"')
    lines.append("")
    lines.extend(_forward_declarations(registry.classes))
    lines.append("")
    lines.append("namespace RF {")
    lines.append("")

    for class_ir in registry.classes:
        lines.append(f"extern const TypeInfo {class_ir.type_symbol};")

    lines.append("")
    lines.append("template<typename T>")
    lines.append("struct StaticType;")
    lines.append("")
    lines.append("#ifndef KISEKI_CORE_RTTI_RUNTIME_DEFINED")
    lines.append("template<typename T>")
    lines.append("constexpr TypeId TypeOf() {")
    lines.append("    return StaticType<T>::Id();")
    lines.append("}")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    lines.append("#ifndef KISEKI_CORE_RTTI_RUNTIME_DEFINED")
    lines.append("TypeId ParentOf(TypeId id);")
    lines.append("bool IsA(TypeId self, TypeId target);")
    lines.append("uintptr_t ToInteger(TypeId id);")
    lines.append("")
    lines.append("namespace Detail {")
    lines.append("template<typename T>")
    lines.append("auto DynamicTypeOfImpl(const T* object, int) -> decltype(object->GetTypeId()) {")
    lines.append("    return object ? object->GetTypeId() : InvalidType;")
    lines.append("}")
    lines.append("")
    lines.append("template<typename T>")
    lines.append("auto DynamicTypeOfImpl(const T* object, long) -> decltype(RF_GetTypeId(object)) {")
    lines.append("    return object ? RF_GetTypeId(object) : InvalidType;")
    lines.append("}")
    lines.append("} // namespace Detail")
    lines.append("")
    lines.append("template<typename T>")
    lines.append("TypeId DynamicTypeOf(const T* object) {")
    lines.append("    return Detail::DynamicTypeOfImpl(object, 0);")
    lines.append("}")
    lines.append("")
    lines.append("template<typename Target, typename Source>")
    lines.append("Target* DynamicCast(Source* object) {")
    lines.append("    if (!object) {")
    lines.append("        return nullptr;")
    lines.append("    }")
    lines.append("    return IsA(DynamicTypeOf(object), TypeOf<Target>())")
    lines.append("        ? static_cast<Target*>(object)")
    lines.append("        : nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    for class_ir in registry.classes:
        parent_symbol = _parent_symbol(registry, class_ir)
        lines.append("template<>")
        lines.append(f"struct StaticType<{class_ir.qualified_name}> {{")
        lines.append(f"    static constexpr TypeId Id() {{ return &{class_ir.type_symbol}; }}")
        lines.append(f"    static constexpr TypeId Parent() {{ return {parent_symbol}; }}")
        lines.append("};")
        lines.append("")

    lines.append("} // namespace RF")
    lines.append("")
    return "\n".join(lines)


def _render_source(registry: RegistryIR, header_name: str) -> str:
    lines: list[str] = []
    lines.append(f'#include "{header_name}"')
    for include in _source_includes(registry.classes):
        lines.append(f'#include "{include}"')
    lines.append("")
    lines.append("namespace RF {")
    lines.append("")

    for class_ir in registry.classes:
        parent_symbol = _parent_symbol(registry, class_ir)
        lines.append(f"extern const TypeInfo {class_ir.type_symbol} = {{ {parent_symbol}, {class_ir.depth} }};")

    lines.append("")
    lines.append("#ifndef KISEKI_CORE_RTTI_RUNTIME_DEFINED")
    lines.append("TypeId ParentOf(TypeId id) {")
    lines.append("    return id ? id->parent : InvalidType;")
    lines.append("}")
    lines.append("")
    lines.append("bool IsA(TypeId self, TypeId target) {")
    lines.append("    if (!self || !target) {")
    lines.append("        return false;")
    lines.append("    }")
    lines.append("    for (TypeId current = self; current; current = current->parent) {")
    lines.append("        if (current == target) {")
    lines.append("            return true;")
    lines.append("        }")
    lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("uintptr_t ToInteger(TypeId id) {")
    lines.append("    return reinterpret_cast<uintptr_t>(id);")
    lines.append("}")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    lines.append("} // namespace RF")
    lines.append("")

    for class_ir in registry.classes:
        if class_ir.has_rtti_body:
            lines.append(f"RF::TypeId {class_ir.qualified_name}::GetTypeId() const {{")
            lines.append(f"    return RF::TypeOf<{class_ir.qualified_name}>();")
            lines.append("}")
            lines.append("")

    return "\n".join(lines)


def _source_includes(classes: list[ClassIR]) -> list[str]:
    includes = []
    seen = set()
    for class_ir in classes:
        path = Path(class_ir.location.file)
        include = path.as_posix()
        if include not in seen:
            seen.add(include)
            includes.append(include)
    return includes


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
        lines.append(f"class {name};")
    for namespace in sorted(namespaces):
        class_names = " ".join(f"class {name};" for name in sorted(set(namespaces[namespace])))
        lines.append(f"namespace {namespace} {{ {class_names} }}")
    return lines


def _parent_symbol(registry: RegistryIR, class_ir: ClassIR) -> str:
    if not class_ir.direct_base:
        return "InvalidType"
    by_name = {item.qualified_name: item for item in registry.classes}
    parent = by_name[class_ir.direct_base.qualified_name]
    return f"&{parent.type_symbol}"
