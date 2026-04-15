"""
Sandboxed execution of user-defined Python feature code.

Uses RestrictedPython to safely execute user code with only
pandas, numpy, and math available. No file I/O, network, or os access.

User code must define a function with this signature:
    def compute(dog_history: pd.DataFrame, race_context: dict) -> float

The dog_history DataFrame has columns:
    trap, finish_position, finish_time, sectional_time, beaten_distance,
    weight_kg, sp_decimal, race_date, track_id, distance_m, grade,
    race_type, going, num_runners, track_name, track_code

The race_context dict has keys:
    trap, dog_id, sp_decimal, track_id, distance_m, grade, race_date, race_type, track_code
"""

import datetime
import logging
import math
import re
import signal
from typing import Any

import numpy as np
import pandas as pd
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import guarded_unpack_sequence, safer_getattr

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 5  # seconds

# Modules that user-defined feature code is allowed to import.
_SAFE_IMPORT_MODULES = frozenset({
    "math", "re", "datetime", "statistics", "collections", "itertools", "functools",
})

_real_import = __import__


def _safe_import(name, *args, **kwargs):
    """Restricted __import__ that only allows whitelisted safe modules."""
    if name not in _SAFE_IMPORT_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in feature sandbox")
    return _real_import(name, *args, **kwargs)


def _build_safe_globals() -> dict:
    """Build restricted globals with allowed builtins."""
    _globals = safe_globals.copy()

    # Allow safe builtins
    allowed_builtins = {
        "__import__": _safe_import,
        "abs": abs,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        "isinstance": isinstance,
        "type": type,
    }
    _globals["__builtins__"] = allowed_builtins

    # Allow safe modules
    _globals["pd"] = pd
    _globals["np"] = np
    _globals["math"] = math
    _globals["re"] = re
    _globals["datetime"] = datetime

    # Required guards for RestrictedPython
    _globals["_getiter_"] = default_guarded_getiter
    _globals["_getitem_"] = default_guarded_getitem
    _globals["_getattr_"] = safer_getattr
    _globals["_unpack_sequence_"] = guarded_unpack_sequence
    _globals["_iter_unpack_sequence_"] = guarded_unpack_sequence

    # Allow print (captured, not actually printed)
    _globals["_print_"] = lambda *args, **kwargs: None

    return _globals


def _timeout_handler(signum, frame):
    raise TimeoutError(f"Feature code execution exceeded {EXECUTION_TIMEOUT}s timeout")


def execute_feature_code(
    code: str,
    dog_history: pd.DataFrame,
    race_context: dict[str, Any],
) -> tuple[float | None, str | None]:
    """
    Execute user-provided feature code in a sandbox.

    Returns (value, error). If successful, error is None.
    If failed, value is None and error contains the error message.
    """
    # Compile with RestrictedPython
    try:
        byte_code = compile_restricted(code, filename="<feature>", mode="exec")
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Compilation error: {e}"

    if byte_code is None:
        return None, "Failed to compile code"

    _globals = _build_safe_globals()
    _locals: dict = {}

    # Execute the code to define the compute function
    try:
        # Set timeout (only works on Unix)
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(EXECUTION_TIMEOUT)
        except (AttributeError, ValueError):
            # signal.SIGALRM not available on Windows
            pass

        exec(byte_code, _globals, _locals)

        # Call the compute function
        compute_fn = _locals.get("compute")
        if compute_fn is None:
            return None, "Code must define a 'compute(dog_history, race_context)' function"

        result = compute_fn(dog_history, race_context)

        # Cancel timeout
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError, NameError):
            pass

        # Validate result
        if result is None:
            return None, None  # None is a valid return

        try:
            value = float(result)
        except (TypeError, ValueError):
            return None, f"compute() must return a float or None, got {type(result).__name__}"

        if math.isnan(value) or math.isinf(value):
            return None, None  # NaN/inf treated as missing

        return value, None

    except TimeoutError as e:
        return None, str(e)
    except Exception as e:
        # Cancel timeout on error
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        return None, f"Runtime error: {type(e).__name__}: {e}"


def validate_feature_code(code: str) -> str | None:
    """
    Validate feature code without executing it.
    Returns error message or None if valid.
    """
    try:
        byte_code = compile_restricted(code, filename="<feature>", mode="exec")
    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Compilation error: {e}"

    if byte_code is None:
        return "Failed to compile code"

    # Check that 'compute' would be defined
    if "def compute" not in code:
        return "Code must define a 'compute(dog_history, race_context)' function"

    return None
