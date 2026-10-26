from __future__ import annotations

import os
from string import Formatter
from typing import Any, Dict

def extract_format_fields(pattern: str) -> set[str]:
    """
    Return the set of field names required by a `str.format()` pattern.
    Example: "a{seed}_b{p_edge}" -> {"seed", "p_edge"}.
    """
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(pattern):
        if not field_name:
            continue
        # handle "{a.b}" or "{a[0]}" by taking the root name ("a")
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root:
            fields.add(root)
    return fields


def safe_format(
    pattern: str,
    values: Dict[str, Any],
    *,
    aliases: Dict[str, str] | None = None,
) -> str:
    """
    Format `pattern` with `values`, optionally mapping missing keys via `aliases`.
    Raises a KeyError with a helpful message if still missing.
    """
    if aliases:
        for missing_key, existing_key in aliases.items():
            if missing_key not in values and existing_key in values:
                values[missing_key] = values[existing_key]

    required = extract_format_fields(pattern)
    missing = sorted(k for k in required if k not in values)
    if missing:
        available = sorted(values.keys())
        raise KeyError(
            f"Missing format keys {missing} for pattern={pattern!r}. "
            f"Available keys: {available}. "
            f"If this pattern came from another problem, update it or add an alias."
        )

    return pattern.format(**values)

def get_instance_file_name(
    seed: int,
    instance_cnt: int,
    n_nodes: int | None,
    n_scenarios: int | None,
    pattern: str | None = None,
    n_facilities: int | None = None,
    n_customers: int | None = None,
    **extra,
) -> str:
    """Return an instance filename formatted from dimensions, seed and extra fields.

    Use the supplied ``pattern`` or the default and resolve supported aliases;
    raise KeyError when a required format field is missing.
    """
    if pattern is None:
        pattern = "inst_seed{seed}_cnt{instance_cnt}_n{n_nodes}_s{n_scenarios}_p{p_edge}_scen_{scenario_name}.pkl"

    values: dict[str, Any] = dict(
        seed=seed,
        instance_cnt=instance_cnt,
        **extra,
    )
    if n_nodes is not None:
        values["n_nodes"] = n_nodes
    if n_scenarios is not None:
        values["n_scenarios"] = n_scenarios
    if n_facilities is not None:
        values["n_facilities"] = n_facilities
    if n_customers is not None:
        values["n_customers"] = n_customers

    # Backward/compat aliasing: selection-style patterns sometimes use p_select
    # but vertex cover uses p_edge.
    return safe_format(
        pattern,
        values,
        aliases={
            "p_select": "p_edge",
            "m": "n_facilities",
            "n": "n_customers",
        },
    )


def apply_label_variant_suffix(file_name: str, cfg: Any) -> str:
    """Return a label filename with the configured variant suffix before its extension.

    ``problem.data.label_variant`` takes precedence over ``label.prise.output``.
    Empty, None or ``both`` leaves the filename unchanged.
    """
    variant = None
    problem_cfg = cfg.get("problem", None) if hasattr(cfg, "get") else None
    data_cfg = problem_cfg.get("data", None) if (problem_cfg is not None and hasattr(problem_cfg, "get")) else None
    if data_cfg is not None and "label_variant" in data_cfg and data_cfg.get("label_variant") is not None:
        variant = str(data_cfg.get("label_variant")).strip().lower()
    elif "label" in cfg and "prise" in cfg.label and "output" in cfg.label.prise:
        variant = str(cfg.label.prise.output).strip().lower()

    if not variant or variant == "both":
        return file_name

    stem, suffix = os.path.splitext(file_name)
    if stem.endswith(f"_{variant}"):
        return file_name
    return f"{stem}_{variant}{suffix}"
