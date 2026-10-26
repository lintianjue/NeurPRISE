"""Classify label targets and validate their train/validation/test assignments.

Small-scale records can support both training and evaluation. Merged larger
datasets use forward-trace records for training and exact labels for testing;
validate their roles rather than relying only on split sizes.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from typing import Any

ROLE_TRAIN = "train_only"      # forward_trace present, x absent
ROLE_EVAL = "eval_only"        # x present, forward_trace absent
ROLE_BOTH = "both"             # complete record -- the small-scale case
ROLE_NEITHER = "neither"       # neither target: unusable, always a defect


class SplitRoleError(ValueError):
    """A train/val/test split places a record where its label cannot be used."""


def _label_field(record: Any, field: str) -> Any:
    label = record["label"] if isinstance(record, dict) else getattr(record, "label")
    if isinstance(label, dict):
        return label.get(field)
    return getattr(label, field, None)


def record_role(record: Any) -> str:
    """Return the training/evaluation role supported by a record.

    A trace makes it trainable. An explicit x, or an objective without a trace,
    makes it evaluable; the latter supports CFLP labels containing only an optimum.
    """
    has_trace = _label_field(record, "forward_trace") is not None
    has_x = _label_field(record, "x") is not None
    has_obj = _label_field(record, "obj") is not None
    is_eval = has_x or (has_obj and not has_trace)

    if has_trace and is_eval:
        return ROLE_BOTH
    if has_trace:
        return ROLE_TRAIN
    if is_eval:
        return ROLE_EVAL
    return ROLE_NEITHER


def summarize_roles(records: Iterable[Any]) -> dict[str, int]:
    counts = {ROLE_TRAIN: 0, ROLE_EVAL: 0, ROLE_BOTH: 0, ROLE_NEITHER: 0}
    for r in records:
        counts[record_role(r)] += 1
    return counts


def is_trainable(record: Any) -> bool:
    return record_role(record) in (ROLE_TRAIN, ROLE_BOTH)


def is_evaluable(record: Any) -> bool:
    return record_role(record) in (ROLE_EVAL, ROLE_BOTH)


def instance_fingerprint(record: Any) -> str:
    """Return a content hash for checking instance identity and ordering.

    Hash fields in sorted-key order, including array shape, dtype and bytes;
    label contents do not affect the hash.
    """
    instance = record["instance"] if isinstance(record, dict) else getattr(record, "instance")
    data = getattr(instance, "data", instance)
    h = hashlib.md5()
    if isinstance(data, dict):
        for key in sorted(data):
            value = data[key]
            h.update(key.encode())
            tobytes = getattr(value, "tobytes", None)
            if tobytes is not None:
                h.update(str(getattr(value, "dtype", "")).encode())
                h.update(str(getattr(value, "shape", "")).encode())
                h.update(tobytes())
            else:
                h.update(repr(value).encode())
    else:
        h.update(repr(data).encode())
    return h.hexdigest()


def assert_split_roles(
    records: Sequence[Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    context: str = "",
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Validate that split records contain usable training or evaluation targets.

    Raise for unusable records, missing train/validation traces or misplaced test
    labels when evaluable records exist. Trace-only datasets log missing test optima
    instead of raising. Returns the dataset role histogram.
    """
    log = logger or logging.getLogger(__name__)
    where = f" [{context}]" if context else ""
    counts = summarize_roles(records)

    bad_neither = [i for i, r in enumerate(records) if record_role(r) == ROLE_NEITHER]
    if bad_neither:
        raise SplitRoleError(
            f"{len(bad_neither)} record(s) carry neither a forward trace nor a true "
            f"optimum and are unusable{where}; first indices: {bad_neither[:5]}"
        )

    for name, idx in (("train", train_idx), ("val", val_idx)):
        offenders = [i for i in idx if not is_trainable(records[i])]
        if offenders:
            raise SplitRoleError(
                f"{len(offenders)} record(s) in the {name} split have no forward trace"
                f"{where}; first indices: {offenders[:5]}. The split boundary does not "
                f"match the label roles."
            )

    dataset_has_eval = counts[ROLE_EVAL] + counts[ROLE_BOTH] > 0
    offenders = [i for i in test_idx if not is_evaluable(records[i])]
    if offenders:
        message = (
            f"{len(offenders)} of {len(list(test_idx))} record(s) in the test split have "
            f"no true optimum, so no regret can be computed on them{where}; "
            f"first indices: {offenders[:5]}."
        )
        if dataset_has_eval:
            raise SplitRoleError(
                message + " The dataset does contain evaluable records, so this is a "
                "misplaced split boundary."
            )
        # Expected, not a defect: regret on a trace file's test split uses each
        # record's `obj` (the full-problem objective V(Xi)) or a separate BigMILP file.
        log.info(
            "%s This file is forward-trace only: regret uses each record's `obj` "
            "(the full-problem objective) or a separate BigMILP label file.",
            message,
        )

    log.info("Label roles%s: %s", where, counts)
    return counts
