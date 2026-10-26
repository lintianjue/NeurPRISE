import logging

import torch
import torch.nn as nn
from typing import Tuple, Literal

from src.core.types import Instance
from src.models.gnn import ScenarioGCN

# Enums for clean config
FusionMode = Literal["late", "early"]
ContextSource = Literal["det_graph", "learned"]


class GNNBasedEncoder(nn.Module):
    """
    Refactored GNN Encoder with support for Early Fusion (Global Token)
    and MLP / Late Fusion variants.
    """

    def __init__(
            self,
            node_in_dim: int,
            scen_hidden_dim: int,
            scen_out_dim: int,
            det_hidden_dim: int,
            det_out_dim: int,
            num_gcn_layers: int = 2,
            # Set Transformer Config
            set_num_heads: int = 4,
            set_num_layers: int = 2,
            set_ff_dim: int = 256,
            dropout_rate = 0.0,
            # Configuration
            det_context_source: str = "det_graph",  # "det_graph" (GNN) or "learned" (MLP), "scenario_mean"
            fusion_mode: str = "late",  # "late" (Parallel) or "early" (Global Token)
            gnn_type: str = "standard",
            edge_attr_dim: int = 2,
            norm_type: str = "batch",
    ):
        super().__init__()

        # Validations
        assert det_context_source in ["det_graph", "learned", "scenario_mean"], f"Unknown source: {det_context_source}"
        assert fusion_mode in ["late", "early"], f"Unknown fusion: {fusion_mode}"

        self.det_context_source = det_context_source
        self.fusion_mode = fusion_mode
        # Expose input width for checkpoint validation without adding a state-dict entry.
        self.node_in_dim = int(node_in_dim)

        # Reject unknown encoder types instead of silently substituting another architecture.
        _GNN_REGISTRY = {
            "standard": ScenarioGCN,
        }
        if gnn_type not in _GNN_REGISTRY:
            raise ValueError(
                f"Unknown gnn_type={gnn_type!r}. "
                f"Known: {sorted(_GNN_REGISTRY)}."
            )
        GNNClass = _GNN_REGISTRY[gnn_type]
        self.dropout_rate = dropout_rate

        gnn_kwargs = dict(
            node_in_dim=node_in_dim,
            hidden_dim=scen_hidden_dim,
            out_dim=scen_out_dim,
            num_layers=num_gcn_layers,
            dropout=dropout_rate,
            edge_attr_dim=edge_attr_dim,
            norm_type=norm_type,
        )

        self.scenario_encoder = GNNClass(**gnn_kwargs)

        # -------------------------------------------------------
        # 2. Deterministic Encoder (GCN or MLP)
        # -------------------------------------------------------
        if self.det_context_source == "det_graph":
            det_gnn_kwargs = dict(
                node_in_dim=node_in_dim,
                hidden_dim=det_hidden_dim,
                out_dim=det_out_dim,
                num_layers=num_gcn_layers,
                dropout=dropout_rate,
                edge_attr_dim=edge_attr_dim,
                norm_type=norm_type,
            )
            self.det_encoder = GNNClass(**det_gnn_kwargs)

        else:  # "learned" -> Raw MLP encoder
            # Assuming input is [B, node_in_dim] or aggregated features
            # This is a simple projection encoder
            self.det_encoder = nn.Sequential(
                nn.Linear(node_in_dim, det_hidden_dim),
                nn.ReLU(),
                nn.Linear(det_hidden_dim, det_out_dim)
            )

        # -------------------------------------------------------
        # 3. Set Interaction (Transformer)
        # -------------------------------------------------------
        self.set_num_layers = set_num_layers
        if self.set_num_layers <= 0:
            # Ablation: no self-attention over the set (identity mapping)
            self.set_encoder = nn.Identity()
            logging.info("GNNEncoder: Set interaction disabled (set_num_layers=0). Using Identity().")
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=scen_out_dim,
                nhead=set_num_heads,
                dim_feedforward=set_ff_dim,
                batch_first=True,
            )
            self.set_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.set_num_layers
            )

        # -------------------------------------------------------
        # 4. Fusion Adapter (Project Det -> Scen)
        # -------------------------------------------------------
        if self.fusion_mode == "early":
            if det_out_dim != scen_out_dim:
                self.token_proj = nn.Linear(det_out_dim, scen_out_dim)
            else:
                self.token_proj = nn.Identity()
            logging.info(f"GNNEncoder: EARLY FUSION enabled. Source={det_context_source}")
        else:
            self.token_proj = None
            if self.det_context_source == "scenario_mean":
                logging.info("GNNEncoder: NeurPRISE GNN + Transformer; context=mean of pre-Transformer scenario embeddings.")
            else:
                logging.info("GNNEncoder: LATE FUSION. Source=%s", det_context_source)

    def forward(self, instances: Instance, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._infer_device(instances)
        if self.det_context_source == "scenario_mean":
            # A. Get Batch Size efficiently (without running det encoder)
            if "det_graph" in instances.data and "batch" in instances.data["det_graph"]:
                B = instances.data["det_graph"]["batch"].max().item() + 1
            else:
                B = len(instances.data.get("id", [])) if hasattr(instances, "data") else 1

            # B. Encode Scenarios FIRST
            z_sce_raw = self._encode_scenarios(instances, device, explicit_B=B)  # [B, S, D]

            # C. Pool Scenarios to get Context
            z_det = z_sce_raw.mean(dim=1)  # [B, D]

            # D. Apply Set Encoder (Transformer) to Scenarios if configured
            z_sce_refined = self.set_encoder(z_sce_raw)

            return z_det, z_sce_refined
        # STEP 1: Encode Deterministic Context FIRST
        z_det = self._encode_deterministic(instances, device)

        # CAPTURE THE TRUTH
        true_batch_size = z_det.shape[0]

        # STEP 2: Encode Scenarios (Passing the Truth)
        # We tell the helper: "Do not guess. The batch size is X."
        z_sce_raw = self._encode_scenarios(instances, device, explicit_B=true_batch_size)

        # STEP 3: Fusion
        if self.fusion_mode == "early":
            global_token = self.token_proj(z_det)  # [B, D]
            global_token_seq = global_token.unsqueeze(1)  # [B, 1, D]

            # NOW: [12, 1, D] cat [12, S, D] -> SUCCESS
            transformer_input = torch.cat([global_token_seq, z_sce_raw], dim=1)

            z_mixed = self.set_encoder(transformer_input)
            z_sce_refined = z_mixed[:, 1:, :]
            return z_det, z_sce_refined
        else:
            z_sce_refined = self.set_encoder(z_sce_raw)
            return z_det, z_sce_refined

    # --------------------------------------------------------------------------
    # Internal Encoding Logic
    # --------------------------------------------------------------------------

    def _encode_scenarios(self, instances: Instance, device: torch.device, explicit_B: int = None) -> torch.Tensor:
        scenarios = instances.data["scen_graph"]
        x, edge_index, edge_attr, batch = self._to_device(
            device,
            scenarios["x"],
            scenarios["edge_index"],
            scenarios["edge_attr"],
            scenarios["scen_batch"]
        )

        # [Total_Graphs, D]
        z_flat = self.scenario_encoder(
            x=x, edge_index=edge_index, edge_attr=edge_attr, scen_batch=batch
        )

        # Reshape [B, S, D]
        # 1. Determine Batch Size (B)
        B = max(1, explicit_B)  # Safety

        # 2. Determine Sequence Length (S)
        total_graphs = z_flat.size(0)
        S = total_graphs // B

        # 3. Reshape
        return z_flat.view(B, S, z_flat.size(-1))

    def _encode_deterministic(self, instances: Instance, device: torch.device) -> torch.Tensor:
        if self.det_context_source == "learned":
            # MLP Path: Expects a simple feature vector (e.g. aggregated node features)
            # Falling back to 'det_raw' or constructing from graph mean if not present
            if "det_raw" in instances.data:
                raw_input = instances.data["det_raw"].to(device)
            else:
                # Fallback: Use mean of det_graph x if det_raw missing
                g = instances.data["det_graph"]
                # Simple mean pool over nodes for the MLP encoder
                # Note: This is an approximation for testing
                raw_input = self._simple_mean_pool(g, device)

            return self.det_encoder(raw_input)

        else:
            # GNN Path
            g = instances.data["det_graph"]
            x, edge_index, edge_attr, batch = self._to_device(
                device, g["x"], g["edge_index"], g["edge_attr"], g["batch"]
            )
            return self.det_encoder(
                x=x, edge_index=edge_index, edge_attr=edge_attr, scen_batch=batch
            )

    def _simple_mean_pool(self, graph_data, device):
        # Quick helper for MLP fallback input
        x = graph_data["x"].to(device)
        batch = graph_data["batch"].to(device)
        # Scatter mean not implemented here to avoid deps, assuming B=1 or manual loop
        # Use global mean pooling
        return x.mean(dim=0, keepdim=True)

    @staticmethod
    def _to_device(device, *args):
        return [x.to(device) if x is not None else None for x in args]

    @staticmethod
    def _infer_device(instance: Instance) -> torch.device:
        if "det_graph" in instance.data:
            return instance.data["det_graph"]["x"].device
        return torch.device("cpu")
