# Perturb PRISE labels (`perturb_labels.py`)

Script: [`scripts/pipeline/perturb_labels.py`](../../../scripts/pipeline/perturb_labels.py).
Run the commands below from the repository root using the default directories
described in the [reproduction guide](../../REPRODUCE.md).

## Table 18: sensitivity to PRISE supervision

Table 18 takes two steps.

1. **Perturb the labels.** `scripts/pipeline/perturb_labels.py` reads the existing
   VC-20-50 PRISE label file, applies each perturbation to it and writes the result as a new
   label file in `data/`, one file per perturbation (label types
   `prise_noise010`, `prise_noise070`, `prise_permute`, `prise_remap`).
   The original file is not changed. Its fixed random seeds give the paper's perturbed
   labels exactly.
2. **Retrain NeurPRISE on a perturbed file.** The new files are saved in `data/`.
   Choose the perturbed file with `+problem.data.label_type`. Train each file
   with the training seeds of Table 13 (1, 2, 3, 4 and 42): `+train.seed=<seed>` changes only
   the training randomness, so the data file and split stay those of seed 42. The table reports
   the mean and standard deviation over the five models; its "Unperturbed" row is Table 13.

```bash
# step 1: perturbed copies of the PRISE label file
python scripts/pipeline/perturb_labels.py --noise 0.1 0.7 --permute --remap \
    --label data/vertex_cover/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl
# step 2: retrain on one of them (here gain noise, sigma = 0.1, training seed 1)
scripts/reproduce.sh train vertex_cover 20 50 2500 model=det_source_sce_mean \
    losses=score_weighted_loss_flat cuda:0 32 uniform data.num_workers=2 \
    +problem.data.label_type=prise_noise010 +train.seed=1
```

In step 1, `--noise 0.1` produces the suffix `noise010` (`100 × sigma`, padded to
three digits) and writes:

```text
data/vertex_cover/label_prise_noise010_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl
```

In step 2, `+problem.data.label_type=prise_noise010` selects that file by filling
the training filename pattern `label_{label_type}_seed{seed}_...pkl`. The remaining
fields come from the data configuration and must match the generated file;
`+train.seed=1` leaves the data seed at 42. Likewise, `--noise 0.7`, `--permute` and
`--remap` produce files selected by `prise_noise070`, `prise_permute` and
`prise_remap`, respectively.
