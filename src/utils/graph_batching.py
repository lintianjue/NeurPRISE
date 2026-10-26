# src/utils/graph_batching.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import torch

from src.core.problem import IProblem
from src.core.types import Instance


class _GraphLike(Protocol):
    x: torch.Tensor
    edge_index: torch.Tensor


class _ProblemLike(Protocol):
    def to_pyg_data(self, inst: Instance) -> List[_GraphLike]: ...


@dataclass(frozen=True)
class ScenarioGraphBatch:
    x: torch.Tensor                 # [total_nodes, F]
    edge_index: torch.Tensor        # [2, total_edges]
    edge_attr: torch.Tensor         # [total_edges, E_dim]
    scen_batch: torch.Tensor        # [total_nodes] node -> global scenario id
    inst_batch: torch.Tensor        # [B*S] scenario -> instance id
    B: int
    S: int

@dataclass(frozen=True)
class DetGraphBatch:
    """
    Batch object for Deterministic (Master) Graphs.
    """
    x: torch.Tensor  # [total_nodes, F]
    edge_index: torch.Tensor  # [2, total_edges]
    edge_attr: torch.Tensor  # [total_edges, E_dim] <--- Added
    batch: torch.Tensor  # [total_nodes] -> instance_id (0..B-1)
    B: int


def build_batched_scenario_graph(
    per_instances: List[Instance],
    problem: _ProblemLike,
    *,
    expected_S: int,
) -> ScenarioGraphBatch:
    """
    CPU-only. Concatenate all scenarios for all instances into one big graph pack.
    Pure function => easy to unit test.
    """
    B = len(per_instances)
    S = int(expected_S)

    all_x: List[torch.Tensor] = []
    all_e: List[torch.Tensor] = []
    all_ea: List[torch.Tensor] = []
    all_scen_batch: List[torch.Tensor] = []
    scenario_instance_ids: List[int] = []

    node_offset = 0
    global_scen_idx = 0

    for inst_id, inst_single in enumerate(per_instances):
        data_list = problem.to_pyg_data(inst_single)
        if len(data_list) != S:
            raise RuntimeError(f"Expected {S} scenarios, got {len(data_list)} for instance {inst_id}.")

        for g in data_list:
            x_i = g.x
            e_i = g.edge_index
            ea_i = g.edge_attr
            if x_i.dim() != 2:
                raise ValueError(f"graph.x must be 2D [n_nodes,F], got {tuple(x_i.shape)}")
            if e_i.dim() != 2 or e_i.size(0) != 2:
                raise ValueError(f"edge_index must be [2,E], got {tuple(e_i.shape)}")

            n = x_i.size(0)
            all_x.append(x_i)
            all_e.append(e_i + node_offset)
            all_ea.append(ea_i)
            all_scen_batch.append(torch.full((n,), global_scen_idx, dtype=torch.long))
            scenario_instance_ids.append(inst_id)

            node_offset += n
            global_scen_idx += 1

    if not all_x:
        raise RuntimeError("No scenario graphs produced.")

    x = torch.cat(all_x, dim=0).contiguous()
    edge_index = torch.cat(all_e, dim=1).contiguous()
    edge_attr = torch.cat(all_ea, dim=0).contiguous()
    scen_batch = torch.cat(all_scen_batch, dim=0).contiguous()
    inst_batch = torch.tensor(scenario_instance_ids, dtype=torch.long)

    if global_scen_idx != B * S:
        raise RuntimeError(f"Expected B*S={B*S} scenarios, got {global_scen_idx}.")

    return ScenarioGraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        scen_batch=scen_batch,
        inst_batch=inst_batch,
        B=B,
        S=S,
    )




def build_batched_det_graph(
        per_instances: List[Instance],
        problem: IProblem,
) -> DetGraphBatch:
    """
    Concatenate deterministic graphs from B instances into one batched graph.
    """
    B = len(per_instances)

    all_x: List[torch.Tensor] = []
    all_e: List[torch.Tensor] = []
    all_ea: List[torch.Tensor] = []  # <--- Accumulator for edge attributes
    all_batch: List[torch.Tensor] = []

    node_offset = 0

    for inst_id, inst_single in enumerate(per_instances):
        # Fetch deterministic graph (1 per instance)
        det_data = problem.to_pyg_data_det(inst_single)

        x_i = det_data.x
        e_i = det_data.edge_index
        ea_i = det_data.edge_attr  # <--- Capture edge attributes

        # Basic Validation
        if x_i.dim() != 2:
            raise ValueError(f"det_graph.x must be 2D, got {tuple(x_i.shape)}")
        if e_i.dim() != 2 or e_i.size(0) != 2:
            raise ValueError(f"det_graph.edge_index must be [2,E], got {tuple(e_i.shape)}")
        if ea_i.dim() != 2:
            raise ValueError(f"det_graph.edge_attr must be 2D [E,D], got {tuple(ea_i.shape)}")

        n = x_i.size(0)

        all_x.append(x_i)
        all_e.append(e_i + node_offset)
        all_ea.append(ea_i)  # <--- Append

        # Batch index maps nodes to the instance ID (0..B-1)
        # Use same device as x_i to avoid mismatches
        all_batch.append(torch.full((n,), inst_id, dtype=torch.long, device=x_i.device))

        node_offset += n

    if not all_x:
        raise RuntimeError("No deterministic graphs produced.")

    x = torch.cat(all_x, dim=0).contiguous()
    edge_index = torch.cat(all_e, dim=1).contiguous()
    edge_attr = torch.cat(all_ea, dim=0).contiguous()  # <--- Concatenate
    batch = torch.cat(all_batch, dim=0).contiguous()

    return DetGraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        B=B,
    )