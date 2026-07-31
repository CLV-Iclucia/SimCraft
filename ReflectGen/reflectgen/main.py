from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .diagnostics import ReflectionError
from .generator import generate_registry
from .ir import read_unit, write_unit
from .scanner import scan_files
from .validator import merge_units


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reflectgen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan C++ files and emit a reflection unit IR")
    scan_parser.add_argument("--target", required=True)
    scan_parser.add_argument("--source-target")
    scan_parser.add_argument("--input", action="append", default=[])
    scan_parser.add_argument("--input-list", action="append", default=[])
    scan_parser.add_argument("--output-ir", required=True)
    scan_parser.add_argument("--include-dir", action="append", default=[])
    scan_parser.add_argument("--define", action="append", default=[])
    scan_parser.add_argument("--std")
    scan_parser.add_argument("--clang-arg", action="append", default=[])
    scan_parser.add_argument("--compile-commands")

    gen_parser = subparsers.add_parser("generate", help="merge unit IR and emit C++ registry files")
    gen_parser.add_argument("--target", required=True)
    gen_parser.add_argument("--input-ir", action="append", default=[])
    gen_parser.add_argument("--input-ir-list")
    gen_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "scan":
            _run_scan(args)
        elif args.command == "generate":
            _run_generate(args)
    except ReflectionError as exc:
        print(f"reflectgen error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _run_scan(args: argparse.Namespace) -> None:
    inputs = _collect_paths(args.input, args.input_list)
    unit = scan_files(inputs, target=args.target, source_target=args.source_target)
    write_unit(unit, Path(args.output_ir))
    print(f"reflectgen: scanned {len(inputs)} files, found {len(unit.classes)} reflected classes")


def _run_generate(args: argparse.Namespace) -> None:
    ir_paths = [Path(path) for path in args.input_ir]
    if args.input_ir_list:
        ir_paths.extend(_read_list_file(Path(args.input_ir_list)))
    units = [read_unit(path) for path in ir_paths]
    registry = merge_units(args.target, units)
    header, source, ir = generate_registry(registry, Path(args.output_dir))
    print(f"reflectgen: generated {header}")
    print(f"reflectgen: generated {source}")
    print(f"reflectgen: wrote {ir}")


def _collect_paths(inputs: list[str], input_lists: list[str]) -> list[Path]:
    paths = [Path(item) for item in inputs]
    for list_path in input_lists:
        paths.extend(_read_list_file(Path(list_path)))
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _read_list_file(path: Path) -> list[Path]:
    if not path.exists():
        return []
    result: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(Path(stripped))
    return result


if __name__ == "__main__":
    main()

