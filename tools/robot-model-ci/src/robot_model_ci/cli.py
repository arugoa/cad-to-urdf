from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .checks import validate_path
from .compare import compare_dirs


def _write_report(report: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output:
        dest = Path(output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered + "\n", encoding="utf-8")
        print(f"JSON report written to {dest}")
    print(rendered)


def _summary(report: dict[str, Any], is_compare: bool = False) -> None:
    summary = report.get("summary", {})
    if is_compare:
        print(f"Model comparison: {report.get('status')} | breaking={summary.get('breaking_changes', 0)} "
              f"warnings={summary.get('warnings', 0)} info={summary.get('info', 0)}")
        for change in report.get("changes", []):
            print(f"[{change['severity']}] {change['code']}: {change['message']}")
    else:
        print(f"Model validation: {report.get('status')} | models={summary.get('models', 0)} "
              f"errors={summary.get('errors', 0)} warnings={summary.get('warnings', 0)}")
        for issue in report.get("issues", []):
            print(f"[{issue['severity']}] {issue['file']} {issue['code']}: {issue['message']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-model-ci", description="Validate and compare URDF/MJCF artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate URDF/MJCF files and referenced local assets")
    validate.add_argument("target", help="model file or generated model directory")
    validate.add_argument("--json", dest="json_path", help="write JSON report to this path")
    compare = sub.add_parser("compare", help="compare candidate artifacts against a baseline")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--json", dest="json_path", help="write JSON report to this path")
    compare.add_argument("--tolerance", type=float, default=1e-6, help="numeric comparison tolerance (default: 1e-6)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = validate_path(args.target)
        _summary(report)
        if args.json_path:
            Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"JSON report written to {args.json_path}")
        return 0 if report.get("status") == "passed" else 1
    try:
        report = compare_dirs(args.baseline, args.candidate, tolerance=args.tolerance)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _summary(report, is_compare=True)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON report written to {args.json_path}")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
