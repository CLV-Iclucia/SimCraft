# Phase 1: pybind11 Build Infrastructure - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the CMake/pybind11 foundation so that a `simcraft` Python module compiles, links against the existing C++ libraries, and imports in Python. Phase 1 delivers a working `.pyd` with a placeholder function (version string) — no actual binding logic.

</domain>

<decisions>
## Implementation Decisions

### pybind11 Integration
- **D-01:** pybind11 acquired via **vcpkg** (add `"pybind11"` to `vcpkg.json`). Consistent with existing dependency management (tbb, eigen3, gtest, glfw3, glad all via vcpkg). CMake uses `find_package(pybind11 REQUIRED)`.

### Directory Structure
- **D-02:** Binding code lives in a new **top-level `python/` directory**. Structure:
  ```
  python/
  ├── CMakeLists.txt
  ├── src/
  │   └── module.cc       (Phase 1: placeholder)
  └── tests/
      └── test_import.py  (Phase 1: import verification)
  ```
  Root CMakeLists.txt adds `add_subdirectory(python)`. The directory name `python/` signals "interface layer" distinct from the PascalCase functional modules (Core, FEM, etc.).

### Python Version Constraint
- **D-03:** CMake minimum is **Python 3.8** (`find_package(Python3 3.8 REQUIRED ...)`). Official documentation will note support for 3.12+. This allows current dev environment (3.10) to work without upgrading. Strict version enforcement deferred to CI/release.

### Link Scope
- **D-04:** Phase 1 `simcraft` target links **FEM** (which transitively brings Core, Maths, Spatify, Deform). The placeholder module won't call FEM code but this validates the entire static-library → .pyd link chain early. If there are ABI issues, symbol conflicts, or missing exports, Phase 1 catches them instead of Phase 2+.

### Build Type (clarified)
- **D-05:** All existing C++ libraries remain **static** (.lib). The `simcraft.pyd` is the only shared library in the project — pybind11 bundles all static deps into a single .pyd. No DLL conversion needed.

### Claude's Discretion
- CMake integration details (`pybind11_add_module` vs `add_library` + manual setup)
- Exact `find_package` component lists for Python
- Test runner choice (pytest vs raw python -c)
- `.pyd` output directory placement within build tree

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Build System
- `CMakeLists.txt` — Root CMake; shows existing module inclusion pattern and vcpkg integration
- `vcpkg.json` — Current dependency manifest; add pybind11 here
- `FEM/CMakeLists.txt` — Example of library target + test setup; model for python/ CMakeLists.txt

### Project Context
- `.planning/PROJECT.md` — Key decisions table (pybind11 confirmed, pure CMake, module name, etc.)
- `.planning/REQUIREMENTS.md` — BUILD-01, BUILD-02, BUILD-03 definitions
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (4 items)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **vcpkg manifest (`vcpkg.json`)**: Already works; just append `"pybind11"` to dependencies array
- **Module CMakeLists.txt pattern (FEM/)**: `file(GLOB_RECURSE srcs ...)`, `add_library`, `target_link_libraries`, `target_include_directories` — same skeleton for `python/`

### Established Patterns
- **Static library modules**: All modules are `STATIC` libraries with `PUBLIC` include dirs. FEM links Maths/Core/Spatify/Deform transitively — single `target_link_libraries(simcraft PRIVATE FEM)` pulls everything.
- **Test pattern**: `enable_testing()` + `foreach` over test files. Python tests will need a different approach (CTest with Python interpreter).
- **MSVC preprocessor**: `/Zc:preprocessor` already set globally — pybind11 is compatible.

### Integration Points
- Root `CMakeLists.txt` line ~30: Insert `add_subdirectory(python)` after Apps
- `vcpkg.json` dependencies array: Append `"pybind11"`
- Build output: `.pyd` will appear in `build/python/` or CMake binary dir

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for pybind11 module setup.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-pybind11 Build Infrastructure*
*Context gathered: 2026-05-25*
