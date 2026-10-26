# NeurPRISE

Code for **Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete
Uncertainty** (NeurIPS 2026).

This release focuses on NeurPRISE training and evaluation. PRISE is included to
generate its supervision labels.

A two-stage robust optimization (2RO) problem with a discrete uncertainty set Ξ is solved
exactly by one MILP that holds a copy of the recourse for every scenario, so its cost
grows with |Ξ|. Scenario reduction solves the problem on a small subset R ⊂ Ξ instead.

* **PRISE** (PRoblem-drIven SEquential lookahead) builds R greedily: at each step it
  adds the scenario whose inclusion raises the reduced objective V(R) the most. Each step
  solves one reduced MILP per candidate scenario, so it is accurate but expensive.
* **NeurPRISE** learns to imitate PRISE. A GNN encodes every scenario's MILP as a
  bipartite graph, a Transformer relates the scenarios to one another, and an attention
  decoder scores all of them in a single forward pass. The top-k scenarios form R.

Problems: Selection (SEL), Vertex Cover (VC), and Capacitated Facility Location (CFLP,
with uncertainty in the constraints).

## Install

```bash
conda create -n neurprise python=3.12 && conda activate neurprise
pip install -r requirements.txt          # Gurobi: bring your own license (academic licenses are free)
```

Activate `neurprise` in each new terminal. If your Gurobi license is outside the
standard location, set its path before running experiments:

```bash
export GRB_LICENSE_FILE="/absolute/path/to/gurobi.lic"   # replace with your license path
python -c 'import gurobipy as gp; env = gp.Env(); env.dispose()'   # verify authorization
```

The license must be valid for the machine running the code. A `HostID mismatch`
means a machine-bound license was generated for another machine and must be regenerated
for this one. Keep the license file outside the repository.

## Quick start

```bash
scripts/reproduce.sh download   # NeurPRISE checkpoints, instances, labels and objective cache
scripts/reproduce.sh eval --table 1 --threads 8   # local run with 8 Gurobi threads
```

`scripts/reproduce.sh eval --table 2 --threads 8` reruns NeurPRISE evaluations for that table from the
released checkpoints and saves new per-instance results and summary statistics. It does
not require saved paper results or compare against printed table values. Rerun results may
differ slightly from the paper's (hardware, software versions, thread counts, randomness);
exact reproduction is not guaranteed.
`--threads` sets the Gurobi thread count in both evaluation stages.
[doc/REPRODUCE.md](doc/REPRODUCE.md) covers every mode and how long each takes.

## Repository layout

```
src/
  core/         Instance, Labels, ForwardTrace data types; the IProblem interface
  problems/     SEL, VC, CFLP: instance sampling, exact MILP, graph encoding;
                scenario distributions (uniform, normal, multimodal)
  heuristics/   PRISE supervision (StepwiseGreedySearch over an Incremental2ROSolver)
  models/       NeurPRISE: GNN + Transformer encoder, attention set decoder
  training/     DataModule, losses, Trainer
  eval/         out-of-sample regret evaluation
  utils/        run-directory I/O, label files, solver helpers
scripts/
  01_create_instance.sh, 02_label.sh, 03_train.sh, 04_05_eval_ml_2ro_e2e.sh
                   pipeline stages (see below)
  reproduce.sh     single entry point for reproducing the paper
  eval/            regret evaluators that write the result JSONs
  pipeline/        label merging and checks, checkpoint provenance
configs/         Hydra configs (problem, model, loss, train, eval)
repro/           runnable experiment recipes, checkpoint configurations and command generator
tests/           pytest suite
```

Pipeline stages (run from the repository root):

| Stage | Script | Output |
|---|---|---|
| 1. instances | `scripts/01_create_instance.sh` | `data/<problem>/inst_*.pkl` |
| 2. PRISE labels | `scripts/02_label.sh` | `data/<problem>/label_prise_*.pkl` |
| 3. training | `scripts/03_train.sh` | a NeurPRISE checkpoint |
| 4. selection + reduced solve | `scripts/04_05_eval_ml_2ro_e2e.sh` | selected scenarios per test instance |
| 5. regret | `scripts/eval/eval_regret.py` | `eval_regret/<problem>/<tag>_<method>_k<k>.json` |

[doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) explains the classes and the data flow
between these stages.

No directory environment variables are needed. Paths are relative to the repository:

| Content | Default location |
|---|---|
| Downloaded or generated instances and labels | `data/<problem>/` |
| Released models | `checkpoints/` |
| New trained models | `runs/03_train/<run>/model/` |
| Evaluation results | `runs/repro/<recipe>/eval_regret/` |
| Logs | `runs/logs/<stage>/<run>/` |

Data generation, labelling and training use the same `data/` folder automatically.
Existing dataset files are never overwritten: reuse them or choose a different dataset
configuration. Each training run prints its model checkpoint path.
Optional directory overrides are described in [doc/REPRODUCE.md](doc/REPRODUCE.md#custom-directories).

## Citation

If you use this code or build on this work, please cite our paper:

```bibtex
@inproceedings{lin2026neurprise,
  title     = {Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty},
  author    = {Lin, Tianjue and Zhou, Jianan and Bi, Jieyi and Wu, Yaoxin and Song, Wen and Cao, Zhiguang and Zhang, Jie},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Disclaimer

This is research code, provided "as is" and without warranty of any kind (see the
license). You are responsible for your own software environment, solver licenses and
compute costs. Check any result yourself before you rely on it. Rerun results may
differ from the paper's; see [doc/REPRODUCE.md](doc/REPRODUCE.md#5-no-guarantee-of-identical-results).

## License

Apache 2.0, see [LICENSE](LICENSE).
