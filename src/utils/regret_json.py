"""Read legacy regret JSON and write strict JSON with non-finite values preserved as status.

Writers encode non-finite values as null with a companion ``<field>_status``;
readers restore floats and also accept legacy Infinity/NaN literals.
Statuses are finite, infeasible (+inf), undefined (NaN) and neg_infinite (-inf).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, Optional

SCHEMA_VERSION = "regret/1.1"

STATUS_FINITE = "finite"
STATUS_INFEASIBLE = "infeasible"
STATUS_UNDEFINED = "undefined"
STATUS_NEG_INFINITE = "neg_infinite"

_STATUS_SUFFIX = "_status"

#: Numeric fields of a per-instance regret record that may legally be
#: non-finite and therefore get a companion ``<field>_status`` entry.
NONFINITE_FIELDS = ("regret", "eval_on_xi", "reduced_obj", "full_obj")


def classify(value: Any) -> str:
    """Map a numeric value to its status token."""
    if value is None:
        return STATUS_UNDEFINED
    try:
        v = float(value)
    except (TypeError, ValueError):
        return STATUS_UNDEFINED
    if math.isnan(v):
        return STATUS_UNDEFINED
    if math.isinf(v):
        return STATUS_INFEASIBLE if v > 0 else STATUS_NEG_INFINITE
    return STATUS_FINITE


def to_strict(value: Any) -> Optional[float]:
    """``None`` for any non-finite value, else the float itself."""
    return float(value) if classify(value) == STATUS_FINITE else None


def from_status(value: Any, status: Optional[str]) -> float:
    """Inverse of :func:`to_strict` — rebuild the float a consumer expects."""
    if value is not None:
        return float(value)
    if status == STATUS_INFEASIBLE:
        return math.inf
    if status == STATUS_NEG_INFINITE:
        return -math.inf
    return math.nan


def record_value(record: Dict[str, Any], field: str) -> float:
    """Return a numeric field as a float from either supported JSON encoding.

    Restore null values using ``<field>_status``; legacy Infinity/NaN remain supported.
    """
    return from_status(record.get(field), record.get(field + _STATUS_SUFFIX))


def annotate_record(record: Dict[str, Any],
                    fields: Iterable[str] = NONFINITE_FIELDS) -> Dict[str, Any]:
    """Return a copy of ``record`` in the strict encoding (null + status)."""
    out = dict(record)
    for field in fields:
        if field not in out:
            continue
        status = classify(out[field])
        out[field] = to_strict(out[field])
        out[field + _STATUS_SUFFIX] = status
    return out


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Used for aggregate/summary values that have no companion status field
    (means over an all-infeasible cell, for instance).
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def dump_strict_fh(obj: Any, fh, indent: int = 2) -> None:
    """Write strict RFC-8259 JSON to an already-open file handle.

    Raises ``ValueError`` (via ``allow_nan=False``) if anything non-finite
    survived sanitization, which would mean a caller bypassed the encoding.
    """
    json.dump(_sanitize(obj), fh, indent=indent, allow_nan=False)


def dump_strict(obj: Any, path: str, indent: int = 2) -> None:
    """Write strict RFC-8259 JSON: no ``Infinity``/``NaN`` literals, ever."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        dump_strict_fh(obj, f, indent)


def dump_strict_new(obj: Any, rp, rel_path, indent: int = 2):
    """Exclusive-create write of strict JSON into a run directory.

    ``rp`` is a ``src.utils.run_io.RunPaths``; ``rel_path`` is relative to
    ``rp.run_dir``. Raises ``FileExistsError`` if it already exists (never
    overwrites). Returns the final ``Path``.
    """
    from src.utils.run_io import publish_new
    return publish_new(rp, rel_path, lambda fh: dump_strict_fh(obj, fh, indent), mode="w")


def load_regret_json(path: str) -> Dict[str, Any]:
    """Load a regret artifact written in either encoding.

    Legacy ``Infinity``/``NaN`` literals are parsed by Python's own decoder;
    no normalization is applied to the returned structure, so the caller sees
    the file as written. Use :func:`record_value` to read numeric fields
    encoding-agnostically.
    """
    with open(path) as f:
        return json.load(f)


def is_strict_file(path: str) -> bool:
    """True if the file contains no non-standard JSON constants."""
    with open(path) as f:
        text = f.read()
    try:
        json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        return False
    return True


def _reject_constant(name: str):  # pragma: no cover - trivial
    raise ValueError(f"non-standard JSON constant: {name}")
