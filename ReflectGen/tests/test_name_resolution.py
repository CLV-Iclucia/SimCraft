from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflectgen.diagnostics import ReflectionError
from reflectgen.ir import (  # noqa: E402
    ClassIR,
    DirectBaseIR,
    SourceLocation,
    UnitIR,
)
from reflectgen.validator import merge_units  # noqa: E402


def _class(
    name: str,
    qualified_name: str,
    namespace: str,
    base: str | None = None,
) -> ClassIR:
    return ClassIR(
        kind="struct",
        name=name,
        qualified_name=qualified_name,
        namespace=namespace,
        location=SourceLocation("test.h", 1, 1),
        direct_base=DirectBaseIR(base) if base else None,
    )


def test_resolves_relative_qualified_base_name() -> None:
    root = _class("SceneObject", "ksk::runtime::SceneObject", "ksk::runtime")
    body = _class(
        "ABDBodyObjectDesc",
        "ksk::scene::ABDBodyObjectDesc",
        "ksk::scene",
        "runtime::SceneObject",
    )

    registry = merge_units("test", [UnitIR("unit", None, [root, body])])

    assert body.direct_base is not None
    assert body.direct_base.qualified_name == "ksk::runtime::SceneObject"
    assert registry.roots == ["ksk::runtime::SceneObject"]


def test_resolves_same_namespace_short_base_name() -> None:
    root = _class("Base", "ksk::scene::Base", "ksk::scene")
    derived = _class("Derived", "ksk::scene::Derived", "ksk::scene", "Base")

    merge_units("test", [UnitIR("unit", None, [root, derived])])

    assert derived.direct_base is not None
    assert derived.direct_base.qualified_name == "ksk::scene::Base"


def test_rejects_ambiguous_suffix_base_name() -> None:
    left = _class("Base", "left::Base", "left")
    right = _class("Base", "right::Base", "right")
    derived = _class("Derived", "other::Derived", "other", "Base")

    with pytest.raises(ReflectionError, match="ambiguous reflected type name Base"):
        merge_units("test", [UnitIR("unit", None, [left, right, derived])])
