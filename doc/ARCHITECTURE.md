# Architecture

This page describes the main classes and how data moves from instance generation to a
number in a paper table. Paths are relative to the repository root.

## Data flow

```
 configs/problem/*.yaml ─┐
                         ▼
 01  IProblem.sample()                     data/<problem>/inst_*.pkl         list[Instance]
                         │
 02  PRISE on each instance                data/<problem>/label_prise_*.pkl  list[{instance, label: Labels}]
     (StepwiseGreedySearch ─ ForwardTrace)
     + exact full-scenario MILP (Labels.obj, Labels.x)
                         │
 03  DataModule ─ ScenarioReducer ─ loss   <run>/model/*_best.pt             NeurPRISE checkpoint
                         │
 04  score test instances, keep top-k      <run>/eval/<name>/e2e_2ro/k_<ratio>/   selected scenario indices
     solve the reduced problem
                         │
 05  regret on all scenarios               eval_regret/<problem>/<tag>_<method>_k<k>.json
```

## Core types (`src/core/`)

| Class | Role |
|---|---|
| `Instance` | One 2RO instance: `data` maps names to arrays (costs, graph, the scenario matrix). Batching stacks the same keys along a new first axis. |
| `Labels` | Solver output for one instance: robust objective `obj` = V(Ξ), first-stage solution `x`, and the PRISE `forward_trace`. |
| `ForwardTrace` | One PRISE run: the scenario chosen at each step (`selected_idx`), the candidates' objective values at each step, the marginal `gains`, and the step at which the gains plateaued (`stop_step`, `step_mask`). |
| `IProblem` | Interface every problem implements: `sample`, `solve_exact`, `collate_fn`, `to_pyg_data` (one bipartite graph per scenario), `get_scenarios`. |

These classes are pickled in the released `inst_*.pkl` and `label_*.pkl` files, so their
module paths are fixed (`tests/test_release_hygiene.py` checks this).

## Problems (`src/problems/`)

`SelectionProblem`, `VertexCoverProblem` and `CFLPProblem` implement `IProblem`. Each
problem has:

* **instance sampling**: `sample()` draws the deterministic data and asks a scenario
  generator (`scenario_generator.py`: uniform, normal, multimodal) for the scenario
  matrix.
* **exact solvers**: `solve_robust_bigmilp` is the deterministic-equivalent MILP over all
  scenarios and gives V(Ξ). `solve_robust_ccg` is column-and-constraint generation.
* **incremental model**: `build_base_model` and `add_scenario_to_model` build the reduced
  MILP one scenario at a time. PRISE needs this.
* **graph encoding**: `to_pyg_data` gives one bipartite graph per scenario. The paper
  uses constraint–variable graphs for SEL and VC. For CFLP, `CFLPEntityBipartiteProblem`
  (config `cflp_ebp`) uses a facility–customer graph. The constraint–variable CFLP
  encoding (`CFLPProblem`, config `cflp_cv`, same instances and labels) is the encoding
  ablation. The trainer logs each epoch's time and peak GPU memory.

Each Hydra problem config (`configs/problem/<name>.yaml`) names its class under
`module:` and its scenario generator under `_target_:`.

## PRISE (`src/heuristics/`)

* `Incremental2ROSolver` holds a Gurobi model with the first-stage variables and the
  scenarios committed so far. `evaluate_subset` copies it, adds candidate scenarios and
  solves. A solve that stops on the time limit returns an upper bound, not V(R), so it is
  retried and otherwise excluded. Otherwise the arg-max would favour the candidate whose
  solve failed worst.
* `StepwiseGreedySearch.search(K)` runs Algorithm 1 of the paper: at every step it
  evaluates every remaining scenario, commits the arg-max, records the gain, and stops
  when the gain stays below `eps_gain` for `stop_M` steps. It returns a `ForwardTrace`.
* `eval.py` provides `_compute_effective_K` and `_forward_with_trace` for PRISE
  supervision during label generation.

`scripts/02_prise_label.py` runs PRISE on training instances and the exact MILP on
evaluation instances. At medium and large scale these are two different sets (indices
0–449 and 450–499). `scripts/pipeline/merge_label_roles.py` joins them into one
500-record `*_merged500.pkl`, and `src/utils/label_roles.py` checks that every record
of the train/val/test split carries the label that split needs.

## NeurPRISE model (`src/models/`)

```
Instance ── problem.to_pyg_data ──► S bipartite graphs
                                        │  GNNBasedEncoder
                                        │    ScenarioGCN: 2 GINE layers + mean pool   h_j
                                        │    Transformer encoder over {h_j}           Z_sce  [B,S,d]
                                        │    context z_inst = mean_j h_j               [B,d]
                                        ▼
                               AttentionSetDecoder
                                 H heads: s_hj = <z_inst W_h^Q, Z_sce,j W_h^K>/sqrt(d_k)
                                 MLP over heads ─► logit z_j                            [B,S]
                                        ▼
                               top-k of the logits = reduced scenario set
```

`ScenarioReducer` wires an encoder to a decoder and returns `{"scores", "logits"}`. The
model configs in `configs/model/` choose the parts:

| Config | Used for |
|---|---|
| `det_source_sce_mean` | NeurPRISE (context = mean of the scenario embeddings) |
| `one_shot_crossattn_edge_dim1_small` | architecture ablation: context from the deterministic graph |
| `config_early_fusion_mlp` | architecture ablation: early fusion + MLP decoder |

## Training (`src/training/`)

* `DataModule` loads a label file, normalizes the features per instance
  (`train_scale_norm`), and splits the instances 80/10/10 by index.
* Losses (`configs/losses/`):
  * `score_weighted_loss_flat` is the paper's loss, a KL divergence between the model's
    softmax over scenarios and a target built from the PRISE gains (log-compressed,
    temperature 5).
  * `log_bce_loss` and `multilabel_bce_loss` are the loss ablation.
* `Trainer.fit` runs AdamW with early stopping on the validation loss. The checkpoint name
  records the config, model and loss (`…_c-train_<cfg>_m-<model>_l-<loss>_best.pt`).
  In the release a checkpoint is renamed `checkpoints/<experiment>/<tag>.pt`, and its
  config, model and loss are recorded in `repro/checkpoints.json`; `repro/runs.py:ckpt_overrides`
  turns them into the evaluator's Hydra overrides.

## Evaluation and regret (`src/eval/regret.py`, `scripts/eval/`)

The metric in every table is the out-of-sample regret. Solve the reduced problem on the
k selected scenarios to get x\*, then evaluate x\* against **all** scenarios with optimal
recourse:

    regret % = 100 · (eval_obj(x*) − V(Ξ)) / V(Ξ)

V(Ξ) comes from the exact labels (cached in `eval_regret/full_objs_cache.json`). If the
recourse is infeasible in some scenario (possible for CFLP), `eval_obj` is +∞. The
instance is then counted in `n_infeasible` and left out of the feasible-only mean.

| Evaluator | Methods |
|---|---|
| `eval_regret.py` | NeurPRISE (reads the selections of an e2e run via `--eval-base`) |
| `eval_regret_ablations.py` | NeurPRISE loss, architecture, seed and data-size variants |

Each writes `eval_regret/<problem>/<tag>_<method>[_k<k>].json` with per-instance values
and summary statistics. The tags are `SEL-20-50`, `VC-50-100`, `CFLP-70-70-50` and so on.
`src/utils/regret_json.py` reads these files.

## Run directories (`src/utils/run_io.py`)

The default workflow requires no directory settings:

* Instances and labels are created in `data/<problem>/` and consumed by the next stage.
* Released models stay in `checkpoints/`; newly trained models go to
  `runs/03_train/<run>/model/`.
* Evaluation results go to `runs/repro/<recipe>/eval_regret/`.
* Hydra logs are saved under `runs/logs/<stage>/<run>/`.

Dataset writes and run creation are exclusive: existing files and run directories
are never overwritten. The shared data writer validates file names and rejects
symlink escapes. Default runs are allowed only in the repository's `runs/` subtree;
other input/output overlap remains forbidden. Custom input roots retain isolated
dataset outputs, as described in [REPRODUCE.md](REPRODUCE.md#custom-directories).

## Reproduction layer (`repro/`)

`runs.json` selects runnable experiments by problem, scale, method and table group.
`checkpoints.json` supplies model paths, training configurations and file hashes.
`runs.py` generates commands that load those inputs and write new evaluation results.
Paper-value comparisons and LaTeX generation are not part of the runtime.
