#!/usr/bin/env python
"""
OnlyNifty AI Agent Skeleton Generator (AST Repo-Map).

Generates a dense, token-efficient Symbol Skeleton Map of classes, methods,
arguments, type annotations, and docstrings across the codebase.

Usage:
    python scripts/generate_repo_map.py             # Entire src/ directory
    python scripts/generate_repo_map.py src/options_engine.py  # Single file
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Optional


def extract_file_skeleton(filepath: Path) -> str:
    """Extracts classes, methods, and functions from a Python file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=str(filepath))
    except Exception as e:
        return f"# Error parsing {filepath}: {e}\n"

    lines = [f"### `{filepath.as_posix()}`"]
    doc = ast.get_docstring(tree)
    if doc:
        summary_line = doc.strip().split("\n")[0]
        lines.append(f"> {summary_line}")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or ""
            c_doc_first = f" — {class_doc.strip().split(chr(10))[0]}" if class_doc else ""
            lines.append(f"\n- **`class {node.name}`**{c_doc_first}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    sig = _format_func_sig(item)
                    f_doc = ast.get_docstring(item) or ""
                    f_summary = f"  *`{f_doc.strip().split(chr(10))[0]}`*" if f_doc else ""
                    lines.append(f"  - `def {sig}` {f_summary}")
        elif isinstance(node, ast.FunctionDef):
            sig = _format_func_sig(node)
            f_doc = ast.get_docstring(node) or ""
            f_summary = f"  *`{f_doc.strip().split(chr(10))[0]}`*" if f_doc else ""
            lines.append(f"- `def {sig}` {f_summary}")

    return "\n".join(lines) + "\n\n"


def _format_func_sig(fn: ast.FunctionDef) -> str:
    args = []
    for a in fn.args.args:
        arg_str = a.arg
        if a.annotation:
            arg_str += f": {ast.unparse(a.annotation)}"
        args.append(arg_str)
    
    ret_str = ""
    if fn.returns:
        ret_str = f" -> {ast.unparse(fn.returns)}"
    return f"{fn.name}({', '.join(args)}){ret_str}"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "src"
    target_path = Path(target)

    if target_path.is_file():
        print(extract_file_skeleton(target_path))
        return

    if not target_path.is_dir():
        print(f"Error: Path {target} does not exist.")
        sys.exit(1)

    print("# OnlyNifty AST Codebase Symbol Map\n")
    py_files = sorted(list(target_path.glob("*.py")))
    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue
        print(extract_file_skeleton(py_file))


if __name__ == "__main__":
    main()
