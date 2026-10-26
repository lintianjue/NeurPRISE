# !/usr/bin/env python3
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from src.training import run_naming
from src.utils.omega_conf import register_omega_conf_custom_resolvers
from src.utils.run_io import RunPaths, open_hydra_run, input_file, mutable_path
from src.utils.seed import seed_everything
from src.training.datamodule import DataModule
from src.training.trainer import Trainer
from src.utils.utils import load_problem
from src.utils.vertex_cover.vertex_cover_utils import (
    apply_label_variant_suffix,
    get_instance_file_name,
)
from src.utils.ddp_trace import enable_ddp_trace
from src.utils.transfer import freeze_scenario_encoder, load_pretrained_into

import logging

logger = logging.getLogger(__name__)


def _resolve_p_for_filename(problem_name: str, n_nodes: int | None, p_select: int | None, p_edge: int | None) -> int | None:
    if str(problem_name).lower() == "selection":
        if n_nodes is None:
            return p_select
        return int(n_nodes) // 2
    return p_edge if p_edge is not None else p_select


def _train_label_file_name(cfg: DictConfig) -> str:
    """The training-side label file name (this chain is characterised against the
    eval-side ``create_problem_module`` / ``create_problem_module_with_data_override``
    chains in ``src/utils/omega_conf.py`` -- see ``tests/utils/test_datamodule_chains_characterised.py``;
    they are recorded, not merged)."""
    seed = cfg.seed
    instance_cnt = cfg.data.get('instance_cnt', 1000)
    inst_cfg = cfg.problem.instance
    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)
    n_scenarios = cfg.problem.instance.n_scenarios
    label_type = cfg.problem.data.get("label_type", "prise")
    p_for_filename = _resolve_p_for_filename(
        str(cfg.problem.name),
        n_nodes,
        getattr(cfg.problem.instance, "p_select", None),
        getattr(cfg.problem.instance, "p_edge", None),
    )
    # Build the shared base name from config (DataModule may derive labels via pattern)
    file_name = get_instance_file_name(
        seed=seed,
        instance_cnt=instance_cnt,
        n_nodes=n_nodes,
        n_scenarios=n_scenarios,
        n_facilities=n_facilities,
        n_customers=n_customers,
        p_edge=p_for_filename,
        scenario_name=inst_cfg.scenario_name,
        pattern=cfg.problem.data.get('inst_spec_label_pattern', None),
        label_type=label_type
    )
    file_name = apply_label_variant_suffix(file_name, cfg)
    return file_name


def _data_normalization_kwargs(cfg: DictConfig) -> dict[str, object]:
    """Return the normalization settings consumed by ``DataModule``."""
    return {
        "normalize": bool(cfg.data.normalize),
        "normalize_mode": str(cfg.data.get("normalize_mode", "minmax")),
    }



def _config_name_fragment(node: DictConfig | None, default: str) -> str:
    return run_naming._config_name_fragment(node, default)


def _hydra_choice_fragment(group: str, default: str) -> str:
    return run_naming._hydra_choice_fragment(group, default)


def _hydra_config_name_fragment(default: str = "") -> str:
    return run_naming._hydra_config_name_fragment(default)


def _resolve_ddp_env() -> tuple[bool, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return True, local_rank, rank, world_size


def _resolve_ddp_find_unused_parameters(cfg: DictConfig) -> bool:
    value = cfg.train.get("ddp_find_unused_parameters", None)
    if value is None:
        return True
    return bool(value)


def _log_ddp_context(
    cfg: DictConfig,
    *,
    ddp_enabled: bool,
    local_rank: int,
    rank: int,
    world_size: int,
    global_batch: int,
    per_rank_batch: int,
) -> None:
    if ddp_enabled:
        device_name = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(
            "DDP enabled: local_rank=%s rank=%s world_size=%s device=%s",
            local_rank,
            rank,
            world_size,
            device_name,
        )
    else:
        logger.info("DDP disabled: single-process training.")

    logger.info(
        "Batching: global_batch_size=%s per_rank_batch_size=%s effective_global_batch_size=%s",
        global_batch,
        per_rank_batch,
        per_rank_batch * (world_size if ddp_enabled else 1),
    )


def _open_training_run(cfg, *, ddp_enabled: bool, rank: int, dist_mod=dist, env=None) -> RunPaths:
    """Rank 0 (or the single process) opens the run; the other ranks receive its run_dir by
    broadcast and never create anything. All ranks return the same RunPaths."""
    if not ddp_enabled:
        return open_hydra_run(cfg, __file__, env=env)
    payload = [None]
    if rank == 0:
        rp = open_hydra_run(cfg, __file__, env=env)
        payload = [(tuple(str(r) for r in rp.input_roots), str(rp.run_dir))]
    dist_mod.broadcast_object_list(payload, src=0)
    roots, run_dir = payload[0]
    return RunPaths(input_roots=tuple(Path(r) for r in roots), run_dir=Path(run_dir))


def _model_dir(cfg, rp: RunPaths) -> Path:
    """train.model_output_dir: "" -> run_dir/model; anything else must lie inside the run dir
    (mutable_path raises otherwise). Writes the resolved absolute path back into cfg so
    run_naming.checkpoint_path(cfg) and the sidecar see the same directory."""
    value = str(cfg.train.model_output_dir or "")
    model_dir = mutable_path(rp, value or "model")
    cfg.train.model_output_dir = str(model_dir)
    return model_dir


def _init_checkpoint_record(rp: RunPaths, path) -> dict | None:
    """None when path is falsy; else {"path": str(path), "realpath": str(input_file(rp, path)),
    "sha256": file_sha256(realpath)} -- input_file raises when the checkpoint is not under an
    input root."""
    if not path:
        return None
    realpath = input_file(rp, path)
    with open(realpath, "rb") as fh:
        sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    return {"path": str(path), "realpath": str(realpath), "sha256": sha256}


@hydra.main(config_path="../configs", config_name="train_normalize", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Working directory: {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Seed
    ddp_enabled, local_rank, rank, world_size = _resolve_ddp_env()
    # DDP note: all ranks must execute collectives in the same order. We keep
    # training/validation in sync by:
    # 1) barrier around validation in Trainer, so ranks don't diverge, and
    # 2) using the raw module (model.module) for rank0-only validation to avoid
    #    DDP broadcasts when other ranks are waiting.
    if ddp_enabled:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if cfg.train.get("ddp_trace", False):
            enable_ddp_trace(logger, rank0_only=False)
        seed_everything(int(cfg.seed) + int(rank))
    else:
        seed_everything(cfg.seed)

    # Seed-replication hook (backward-compatible; inert unless set).
    # `cfg.seed` is overloaded: it seeds the RNG *and* names the dataset file
    # (get_instance_file_name below builds "s{seed}_i{cnt}_..."), so overriding
    # it to get a second seed would look for a dataset that does not exist.
    # `+train.seed=N` reseeds only the RNG, leaving the data file untouched.
    # Splits are deterministic index ranges (datamodule.py:105-112), so the
    # train/val/test partition is identical across seeds and only the weight
    # init and the shuffle order vary — which is what a seed replicate means.
    rng_seed = cfg.train.get("seed", None)
    if rng_seed is not None:
        seed_everything(int(rng_seed))
        logger.info("RNG seed overridden by train.seed=%s (data file still s%s)",
                    int(rng_seed), cfg.seed)

    global_batch = int(cfg.train.batch_size)
    if ddp_enabled:
        if global_batch % world_size != 0:
            raise ValueError(
                f"Global batch size {global_batch} must be divisible by world_size {world_size} for DDP."
            )
        per_rank_batch = global_batch // world_size
    else:
        per_rank_batch = global_batch

    cfg.train.batch_size = per_rank_batch
    if "data" in cfg and "batch_size" in cfg.data:
        cfg.data.batch_size = per_rank_batch

    _log_ddp_context(
        cfg,
        ddp_enabled=ddp_enabled,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        global_batch=global_batch,
        per_rank_batch=per_rank_batch,
    )

    rp = _open_training_run(cfg, ddp_enabled=ddp_enabled, rank=rank)

    # Load problem using Hydra config
    problem_module = cfg.problem.module
    problem_kwargs = OmegaConf.to_container(cfg.problem.instance, resolve=True)
    problem = load_problem(problem_module, problem_kwargs)

    file_name = _train_label_file_name(cfg)

    # Create DataModule (it should yield either Instances or (Instances, Labels) tuples)
    dm = DataModule(
        problem=problem,
        data_path=cfg.problem.data.path,
        problem_name=cfg.problem.name,
        file_name=file_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        split_mode=cfg.data.get("split_mode", "ratio"),
        val_split_number=cfg.data.get("val_split_number", 0),
        test_abs_split_number=cfg.data.get("test_abs_split_number", 0),
        train_split_number=cfg.data.get("train_split_number", 0),
        **_data_normalization_kwargs(cfg),
    )
    overfit_n = cfg.train.get("debug", {}).get("overfit_n", None)
    if overfit_n is not None:
        idx = list(range(int(overfit_n)))
        dm.set_splits(idx, idx, idx)

    # Cross-problem transfer hook (backward-compatible; inert unless both keys are set):
    # load explicit label-budget train split + fixed val/test index files.
    split_dir = cfg.data.get("split_index_dir", None)
    train_idx_file = cfg.data.get("train_index_file", None)
    if bool(split_dir) != bool(train_idx_file):
        raise ValueError("data.split_index_dir and data.train_index_file must be set together")
    if split_dir:
        tr = np.load(os.path.join(split_dir, train_idx_file))
        va = np.load(os.path.join(split_dir, "val_idx.npy"))
        te = np.load(os.path.join(split_dir, "test_idx.npy"))
        dm.set_splits(tr.tolist(), va.tolist(), te.tolist())
        logger.info(
            "Transfer splits: train=%d val=%d test=%d from %s (train_file=%s)",
            len(tr), len(va), len(te), split_dir, train_idx_file,
        )
    if ddp_enabled:
        # Construct the sampler only after the final explicit split is installed.
        train_sampler = DistributedSampler(
            dm.train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        dm.set_train_sampler(train_sampler)
        logger.info(
            "DistributedSampler: num_replicas=%s rank=%s drop_last=%s",
            world_size,
            rank,
            False,
        )

    # Model
    model = instantiate(cfg.model)

    # Cross-problem transfer hook (backward-compatible; inert unless keys are set):
    # (1) initialize from a SEL checkpoint via controlled per-tensor load, then
    # (2) optionally freeze the scenario_encoder (GNN). Both happen before the
    # DDP wrap and before the optimizer is built.
    init_ckpt = cfg.train.get("init_from_checkpoint", None)
    init_record = _init_checkpoint_record(rp, init_ckpt)
    if init_ckpt:
        # VC->SEL drops the source's trailing node-degree column (9 -> 8), the mirror
        # of the SEL->VC zero-fill.  Opt-in: left false, a narrowing mismatch still
        # raises, so every existing run keeps the behaviour it was validated under.
        allow_column_drop = bool(cfg.train.get("init_allow_column_drop", False))
        report = load_pretrained_into(
            model, init_record["realpath"], allow_column_drop=allow_column_drop
        )
        logger.info(
            "Init-from-checkpoint: copied=%d zero_filled=%d column_dropped=%d from %s",
            report["num_copied"], report["num_zero_filled"],
            report.get("num_column_dropped", 0), init_ckpt,
        )
    if cfg.train.get("freeze_encoder", False):
        freeze_scenario_encoder(model)
    if ddp_enabled:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        find_unused = _resolve_ddp_find_unused_parameters(cfg)
        logger.info("DDP find_unused_parameters=%s", find_unused)
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused,
        )

    # Optimizer (only trainable params, so a frozen encoder is truly excluded)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    model_dir = _model_dir(cfg, rp)
    os.makedirs(model_dir, exist_ok=True)

    best_model_path = run_naming.checkpoint_path(cfg)
    logger.info(f"Best model checkpoint path: {best_model_path}")

    # Trainer
    trainer = Trainer(
        model,
        opt,
        metrics=cfg.problem.metrics,
        device=(f"cuda:{local_rank}" if ddp_enabled and torch.cuda.is_available() else cfg.train.get("device", "cuda")),
        max_epochs=cfg.train.max_epochs,
        best_model_path=best_model_path,
        patience=cfg.train.patience,
        grad_clip=cfg.train.grad_clip,
        teacher_mode=cfg.train.teacher_mode,
        grad_debug=cfg.train.get("grad_debug", False),
        log_every_n_steps=cfg.train.get("log_every_n_steps", 50),
        amp_dtype=cfg.train.get("amp_dtype", "off"),
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        log_param_update=cfg.train.get("log_param_update", False),
        is_rank0=(rank == 0 if ddp_enabled else True),
    )

    # Train with supervised loss
    loss_fn = instantiate(cfg.losses)

    sidecar_ok = True
    sidecar_reason: str | None = None
    try:
        trainer.fit(dm, problem, loss_fn)

        # --- provenance sidecar -------------------------------------------------------
        # AFTER fit, so the checkpoint exists and its digest describes something; rank 0
        # only, so DDP ranks do not race on one file. This never raises: a sidecar bug must
        # not discard a finished model, so `emit_loudly` returns a bool and the failure is
        # reported at the end of a run rather than thrown from the middle of one. Without
        # this the checkpoint cannot pass scripts/pipeline/verify_ckpt_provenance.py.
        if (not ddp_enabled) or rank == 0:
            try:
                from src.utils.run_metadata import emit_loudly
                sidecar_ok = bool(emit_loudly(
                    best_model_path,
                    # `+train.arm=<name>` declares which experimental arm this checkpoint IS.
                    # Undeclared => null, and a consumer that expects a specific arm then
                    # fails closed, which is the honest outcome rather than a gap.
                    arm=cfg.train.get("arm", None),
                    label_path=dm.data_file,
                    extra={
                        "cfg_seed": int(cfg.seed),
                        "train_seed": cfg.train.get("seed", None),
                        "batch_size": int(cfg.data.batch_size),
                        "global_batch_size": int(global_batch),
                        "normalize": bool(cfg.data.normalize),
                        "normalize_mode": str(cfg.data.get("normalize_mode", "minmax")),
                        "accumulate_grad_batches": int(cfg.train.get("accumulate_grad_batches", 1)),
                        "lr": float(cfg.train.optimizer.lr),
                        "model_name": cfg.model.get("name", None),
                        "loss_name": cfg.losses.get("name", None),
                        "problem_name": cfg.problem.get("name", None),
                        "output_tag": str(cfg.train.get("output_tag", "") or ""),
                        "world_size": int(world_size) if ddp_enabled else 1,
                        "init_from_checkpoint": init_record,
                    },
                    logger=logger,
                ))
                if not sidecar_ok:
                    sidecar_reason = "emit_loudly reported failure; see the [sidecar] line above"
            except Exception as exc:  # the import, or building `extra`, must not kill a finished run
                sidecar_ok = False
                sidecar_reason = repr(exc)
                logger.error("sidecar emission raised: %r", exc)
    finally:
        if ddp_enabled and dist.is_initialized():
            dist.destroy_process_group()

    # The block above must not THROW -- a sidecar bug may not discard a finished model, and
    # the checkpoint is already on disk by here. But its outcome must not be invisible either:
    # a run that exits 0 with no .run.json is refused by the consumer only AFTER a full
    # evaluation budget is spent. So the checkpoint survives, and the run fails loudly here.
    if not sidecar_ok:
        raise RuntimeError(
            f"provenance sidecar was not written for {best_model_path}: {sidecar_reason}. "
            f"The checkpoint itself is intact on disk, but it cannot pass "
            f"scripts/pipeline/verify_ckpt_provenance.py, so no consumer may evaluate it."
        )

    #trainer.fit_debug(dm, problem, loss_fn)


if __name__ == "__main__":
    register_omega_conf_custom_resolvers()
    main()
