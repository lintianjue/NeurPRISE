# Merge training and test labels (`merge_label_roles.py`)

Script: [`scripts/pipeline/merge_label_roles.py`](../../../scripts/pipeline/merge_label_roles.py).
See the [reproduction guide](../../REPRODUCE.md) for the overall pipeline.

## Medium and large instances

At medium and large scale, the paper labels instances 0–449 with PRISE (training) and
450–499 with the exact MILP (test). `scripts/pipeline/merge_label_roles.py` merges the
two files into the `*_merged500.pkl` file that training and evaluation read.
