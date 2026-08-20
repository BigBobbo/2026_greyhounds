"""The capture agent must run on the Python that ships with macOS.

It is the one file in this repo executed on someone else's computer,
where we don't control the interpreter — macOS still ships Python 3.9,
and a 3.10-only construct fails at import with a traceback rather than a
useful message. These checks caught `str | None` in a signature reaching
a user; they exist so the next one is caught here instead.
"""

import ast
import os

import pytest

AGENT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "agent", "betfair_capture_agent.py",
)

MIN_PY = (3, 9)


@pytest.fixture(scope="module")
def source() -> str:
    with open(AGENT) as f:
        return f.read()


def test_agent_parses_on_oldest_supported_python(source):
    """Catches syntax-level features (match statements, etc.)."""
    ast.parse(source, feature_version=MIN_PY)


def test_future_annotations_is_first_statement(source):
    """Without it, PEP 604 annotations are evaluated at import time and
    raise TypeError on 3.9 — which is exactly how this broke before."""
    tree = ast.parse(source)
    statements = [
        node for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    assert statements, "agent has no statements"
    first = statements[0]
    assert isinstance(first, ast.ImportFrom) and first.module == "__future__", (
        "`from __future__ import annotations` must be the first statement"
    )
    assert any(alias.name == "annotations" for alias in first.names)


def test_no_pep604_unions_outside_annotations(source):
    """`X | None` is fine in an annotation (deferred by the future import)
    but not in code that actually runs on 3.9."""

    class Scan(ast.NodeVisitor):
        def __init__(self):
            self.offenders: list[int] = []

        def _strip(self, node):
            for arg in list(node.args.args) + list(node.args.kwonlyargs):
                arg.annotation = None
            node.returns = None

        def visit_FunctionDef(self, node):
            self._strip(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self._strip(node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            node.annotation = None
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.BitOr):
                self.offenders.append(node.lineno)
            self.generic_visit(node)

    scan = Scan()
    scan.visit(ast.parse(source))
    assert not scan.offenders, (
        f"PEP 604 union evaluated at runtime on line(s) {scan.offenders}; "
        "this raises TypeError on Python 3.9"
    )


def test_agent_uses_only_the_standard_library(source):
    """Setup on a home machine is 'install Python and run it' — a
    third-party import silently breaks that promise."""
    third_party = {"httpx", "requests", "pydantic", "sqlalchemy", "urllib3",
                   "dotenv", "numpy", "pandas"}
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    leaked = imported & third_party
    assert not leaked, f"agent imports non-stdlib package(s): {leaked}"
