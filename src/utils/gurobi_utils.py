import gurobipy as gp
from typing import Any, Dict, Optional


def copy_model_with_payload(master: gp.Model, *, payload_attr: str = "_copy_payload") -> gp.Model:
    """Copy a Gurobi model while preserving its Python metadata.

    Reattach the attribute/value pairs in ``master.<payload_attr>`` to the copy,
    because ``Model.copy()`` does not preserve custom attributes. Returns the copy.
    """
    cand = master.copy()

    payload: Optional[Dict[str, Any]] = getattr(master, payload_attr, None)
    if payload:
        for attr_name, attr_val in payload.items():
            setattr(cand, attr_name, attr_val)
        setattr(cand, payload_attr, dict(payload))  # keep payload available on the copy too

    return cand