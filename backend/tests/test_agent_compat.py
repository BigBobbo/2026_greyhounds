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


@pytest.fixture(scope="module")
def agent():
    """Import the agent module by path (it lives outside the package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture_agent", AGENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Betfair replies in XML or JSON depending on the Accept header and how
# early the request is rejected. Reading only JSON meant a real
# INVALID_APP_KEY surfaced to the user as "no error code".
BETFAIR_XML = (
    "<?xml version='1.0' encoding='utf-8'?><fault><faultcode>Client</faultcode>"
    "<faultstring>ANGX-0007</faultstring><detail>"
    "<exceptionname>APINGException</exceptionname><APINGException>"
    "<errorCode>INVALID_APP_KEY</errorCode></APINGException></detail></fault>"
)


@pytest.mark.parametrize("body,expected", [
    (BETFAIR_XML, "INVALID_APP_KEY"),
    (BETFAIR_XML[:210], "INVALID_APP_KEY"),          # truncated mid-document
    ('{"detail":{"APINGException":{"errorCode":"NO_SESSION"}}}', "NO_SESSION"),
    ("", None),
    ("<html>not an api response</html>", None),
    ("{not valid json", None),
])
def test_error_codes_are_extracted_from_either_format(agent, body, expected):
    assert agent._aping_code(body) == expected


def test_known_error_codes_have_actionable_explanations(agent):
    err = agent.BetfairError(400, "INVALID_APP_KEY", BETFAIR_XML)
    assert str(err) == "HTTP 400 / INVALID_APP_KEY"
    assert "app key" in agent.explain_api_error(err).lower()
    # An unknown code still tells the operator what to do with it.
    unknown = agent.BetfairError(400, "SOME_NEW_CODE", "")
    assert "SOME_NEW_CODE" in agent.explain_api_error(unknown)


def test_data_calls_request_json_responses(source):
    """The Accept header is why errors come back parseable at all."""
    assert '"Accept": "application/json"' in source


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
