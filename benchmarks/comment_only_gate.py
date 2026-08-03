"""Prove a change touched only comments, docstrings and whitespace.

Parses each module in the package and hashes its AST with every docstring
blanked out and line numbers dropped. If the hash is unchanged, no executable
statement moved -- which is the entire claim a readability pass makes, and a
much stronger check than reading the diff.

Docstrings are blanked rather than compared because they live in the AST as the
first expression of a module, class or function, so a rewritten docstring would
otherwise read as a code change. Line numbers are dropped because whitespace
moves them.

    python benchmarks/comment_only_gate.py snapshot before.json   # before editing
    python benchmarks/comment_only_gate.py check    before.json   # after editing

Exit code 1 if any module's executable structure changed. Pair it with the test
suite: this proves nothing MOVED, the goldens prove nothing CHANGED NUMERICALLY.
"""

import ast
import hashlib
import json
import os
import sys

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "chimeraboost")


class _BlankDocstrings(ast.NodeTransformer):
    """Replace every docstring with one constant marker."""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0].value.value = "<docstring>"
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def fingerprint(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    tree = _BlankDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()


def collect():
    return {name: fingerprint(os.path.join(PKG, name))
            for name in sorted(os.listdir(PKG)) if name.endswith(".py")}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("snapshot", "check"):
        print(__doc__)
        return 2

    mode, path = sys.argv[1], sys.argv[2]
    now = collect()

    if mode == "snapshot":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(now, f, indent=2)
        print(f"snapshotted {len(now)} modules -> {path}")
        return 0

    with open(path, encoding="utf-8") as f:
        before = json.load(f)

    changed = [k for k in sorted(set(before) | set(now))
               if before.get(k) != now.get(k)]
    for name in sorted(now):
        print(f"  {'CODE CHANGED' if name in changed else 'same':12s} {name}")

    if changed:
        print(f"\n{len(changed)} module(s) changed executable code: "
              f"{', '.join(changed)}")
        return 1

    print(f"\nall {len(now)} modules: comments, docstrings and whitespace only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
