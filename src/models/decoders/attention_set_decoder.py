import torch
import torch.nn as nn
import math

from src.core.types import DecoderOutput


class LinearSetDecoder(nn.Module):
    """
    Scenario scoring head with optional deterministic query.

    det_mode:
      - "use":               use z_det (global query) normally
      - "ignore_same_params": ignore z_det but keep W_d parameters (parameter-matched ablation)
      - "drop_params":       remove W_d entirely (no-det variant, fewer params)
    """
    def __init__(self, scen_dim: int, det_dim: int, hidden_dim: int, det_mode: str = "use"):
        super().__init__()
        assert det_mode in ("use", "ignore_same_params", "drop_params")
        self.det_mode = det_mode

        self.W_h = nn.Linear(scen_dim, hidden_dim, bias=False)

        if det_mode == "drop_params":
            self.W_d = None
            self._det_dim = det_dim  # keep for shape checking / fallback zeros
        else:
            self.W_d = nn.Linear(det_dim, hidden_dim, bias=False)
            self._det_dim = self.W_d.in_features

        self.v = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, z_det, z_sce, **kwargs) -> DecoderOutput:
        if z_sce is None:
            raise RuntimeError("ScenarioScoringHead expects non-None z_sce of shape [B, S, D].")

        z_sce = z_sce.float()
        B, S, scen_dim = z_sce.shape

        # Flatten scenarios: [B, S, D] -> [B*S, D]
        scen_flat = z_sce.view(B * S, scen_dim)

        device = scen_flat.device
        instance_batch = torch.arange(B, device=device).repeat_interleave(S)  # [B*S]

        # scenario term
        Wh = self.W_h(scen_flat)  # [B*S, h]

        # deterministic term
        if self.det_mode == "drop_params":
            # no W_d term at all
            Wd_full = 0.0
        else:
            det_dim = self._det_dim

            if self.det_mode == "ignore_same_params":
                det_emb = scen_flat.new_zeros((B, det_dim))
            else:  # "use"
                if z_det is None:
                    det_emb = scen_flat.new_zeros((B, det_dim))
                else:
                    if z_det.dim() != 2 or z_det.size(0) != B:
                        raise RuntimeError(f"z_det must have shape [B, det_dim], got {tuple(z_det.shape)}")
                    if z_det.size(1) != det_dim:
                        raise RuntimeError(f"z_det feature dim {z_det.size(1)} != expected det_dim {det_dim}")
                    det_emb = z_det.float()

            Wd_full = self.W_d(det_emb)[instance_batch]  # [B*S, h]

        u = torch.tanh(Wh + Wd_full)         # [B*S, h]
        logits = self.v(u).squeeze(-1).view(B, S)  # [B, S]
        probs = torch.softmax(logits, dim=-1)

        return DecoderOutput(
            scores=logits,
            logits=logits,
            probs=probs,
            aux={"z_sce": z_sce, "det_mode": self.det_mode},
        )


class AttentionSetDecoder(nn.Module):
    """
    Multi-head dot-product query scoring.
    Inputs:
      z_inst: [B, d_det]
      z_sce:  [B, S, d_sce]
    Output:
      logits: [B, S]
    """

    def _init_weights(self):
        # 1. Projections (Wq, Wk): Use Xavier Uniform
        # This keeps the embedding variance stable before the dot product
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

        # 2. Mixer (head_mix):
        if isinstance(self.head_mix, nn.Linear):
            nn.init.xavier_uniform_(self.head_mix.weight)
        elif isinstance(self.head_mix, nn.Sequential):
            for m in self.head_mix:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)


    def __init__(self, det_dim: int, sce_dim: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.H = num_heads
        self.Dh = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.Wq = nn.Linear(det_dim, num_heads * head_dim, bias=False)
        self.Wk = nn.Linear(sce_dim, num_heads * head_dim, bias=False)

        # head-wise scoring vector: one scalar per head from q·k
        # then combine heads into final logit
        self.head_mix = nn.Sequential(
            nn.Linear(num_heads, num_heads * 4),  # Standard expansion ratio is 4x
            nn.ReLU(),
            nn.Linear(num_heads * 4, 1)  # Project to scalar decoders
        )
        self._init_weights()

    def forward(self, z_inst: torch.Tensor, z_sce: torch.Tensor, **kwargs) -> DecoderOutput:
        B, S, _ = z_sce.shape

        q = self.Wq(z_inst).view(B, self.H, self.Dh)              # [B,H,Dh]
        k = self.Wk(z_sce).view(B, S, self.H, self.Dh)            # [B,S,H,Dh]

        # per-head compatibility per scenario: [B,S,H]
        compat = (k * q.unsqueeze(1)).sum(dim=-1) * self.scale

        # combine heads -> logits [B,S]
        logits = self.head_mix(compat).squeeze(-1)

        probs = torch.softmax(logits, dim=-1)

        return DecoderOutput(
            scores=logits,
            logits=logits,
            probs=probs,
            aux={"z_sce": z_sce, "compat": compat},
        )


class MLPSetDecoder(nn.Module):
    """
    Ablation Decoder: MLP-based scoring for Early Fusion.
    Assumes deterministic context is already integrated into z_sce by the encoder.
    """

    def __init__(self, sce_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Simple MLP to score each scenario independently
        self.mlp = nn.Sequential(
            nn.Linear(sce_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_inst: torch.Tensor, z_sce: torch.Tensor, **kwargs) -> DecoderOutput:
        # z_inst (det_dim) is ignored as it is already fused in the encoder
        B, S, _ = z_sce.shape

        # Flatten scenarios to process through shared MLP: [B*S, D]
        logits = self.mlp(z_sce).squeeze(-1)  # [B, S]

        probs = torch.softmax(logits, dim=-1)

        return DecoderOutput(
            scores=logits,
            logits=logits,
            probs=probs,
            aux={"z_sce": z_sce},
        )
