# src/models/scenario_reducer.py
from src.core.problem import IProblem
from src.core.types import DecoderOutput
from src.utils.utils import print_param_counts
from typing import Optional, Any, Dict
import torch
import torch.nn as nn

class ScenarioReducer(nn.Module):
    """Encode instances and scenarios, then decode scenario scores.

    Forward returns a score dictionary with [B, S] summary logits; stepwise
    decoders receive T and may store per-step logits in ``DecoderOutput.aux``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        name: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

        print_param_counts(self, name="ScenarioReducer")

    def _normalize_decoder_out(self, decoder_out: Any) -> DecoderOutput:
        if isinstance(decoder_out, DecoderOutput):
            return decoder_out

        if isinstance(decoder_out, dict):
            scores_raw = decoder_out.get("scores", None)
            if scores_raw is None:
                scores_raw = decoder_out.get("x", None)
            if scores_raw is None:
                raise RuntimeError(
                    "Dict decoder_out must contain 'scores' or legacy 'x'. "
                    f"Got keys: {list(decoder_out.keys())}"
                )

            logits = decoder_out.get("logits", None)
            if logits is None:
                logits = scores_raw

            return DecoderOutput(
                scores=scores_raw,
                logits=logits,
                probs=decoder_out.get("probs", None),
                aux=decoder_out.get("aux", None),
            )

        raise RuntimeError(f"Decoder must return DecoderOutput or dict, got {type(decoder_out)}")

    def forward(
        self,
        instance,
        problem: IProblem,
        *,
        teacher_idx: Optional[Any] = None,
        K: Optional[int] = None,     # selection size (TopK)
        T: Optional[int] = None,     # step count for stepwise decoder (NOT selection size)
        return_aux: bool = False,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decoder_kwargs = decoder_kwargs or {}

        # 1) Encode
        z_inst, z_sce = self.encoder(instance, problem=problem)

        # 2) Score (pass T through; accept extra kwargs)
        decoder_out_raw = self.decoder(
            z_inst,
            z_sce,
            instance=instance,
            problem=problem,
            T=T,
            **decoder_kwargs,
        )

        decoder_out = self._normalize_decoder_out(decoder_out_raw)

        scores_raw = decoder_out.scores
        logits = decoder_out.logits if decoder_out.logits is not None else decoder_out.scores

        # shape check: reducer interface expects [B,S] here
        if scores_raw.dim() != 2:
            raise RuntimeError(f"Expected decoder scores [B,S], got {tuple(scores_raw.shape)}")

        out: Dict[str, Any] = {
            "scores": scores_raw,
            "logits": logits,
            "idx": None,
            "K": None,
        }
        return out


class StepwiseScenarioReducer(nn.Module):
    """
    Stepwise reducer:
      - encoder: returns (z_inst, z_sce)
      - decoder: returns dict with logits_steps, pick_idx_steps, ...
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        name: str,
        side_feature_key: Optional[str] = None,
        side_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.side_feature_key = side_feature_key
        self.side_feature_dim = (
            None if side_feature_dim is None else int(side_feature_dim)
        )
        if self.side_feature_key is None and self.side_feature_dim is not None:
            raise ValueError("side_feature_dim requires side_feature_key.")
        if self.side_feature_key is not None and (
            self.side_feature_dim is None or self.side_feature_dim <= 0
        ):
            raise ValueError(
                "A positive side_feature_dim is required when side_feature_key is set."
            )
        print_param_counts(self, name="StepwiseScenarioReducer")

    def _append_side_features(self, instances, z_sce):
        if self.side_feature_key is None:
            return z_sce
        if self.side_feature_key not in instances.data:
            raise KeyError(
                f"Instance is missing side feature {self.side_feature_key!r}."
            )
        side = instances.data[self.side_feature_key]
        if not hasattr(side, "shape") or len(side.shape) != 3:
            raise RuntimeError(
                f"{self.side_feature_key} must have shape [B,S,W], "
                f"got {getattr(side, 'shape', None)}."
            )
        if tuple(side.shape[:2]) != tuple(z_sce.shape[:2]):
            raise RuntimeError(
                f"{self.side_feature_key} batch/scenario shape "
                f"{tuple(side.shape[:2])} != encoded shape {tuple(z_sce.shape[:2])}."
            )
        if int(side.shape[-1]) != self.side_feature_dim:
            raise RuntimeError(
                f"{self.side_feature_key} width {int(side.shape[-1])} != "
                f"declared side_feature_dim={self.side_feature_dim}."
            )
        side = side.to(device=z_sce.device, dtype=z_sce.dtype)
        return torch.cat([z_sce, side], dim=-1)

    def forward(
        self,
        instances,
        problem: IProblem,
        K: Optional[int] = None,
        T: Optional[int] = None,
        teacher_idx: Optional[Any] = None,
        return_aux: bool = False,
        beam_train: bool = False,
        beam_size: int = 4,
        beam_expand: int = 4,
        length_norm_alpha: float = 0.0,
        mix_teacher_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        z_inst, z_sce = self.encoder(instances, problem=problem)
        z_sce = self._append_side_features(instances, z_sce)
        if T is None:
            raise ValueError("StepwiseScenarioReducer requires T.")
        if not (1 <= int(T) <= int(z_sce.shape[1])):
            raise ValueError(
                f"Stepwise T must satisfy 1 <= T <= S={z_sce.shape[1]}, got {T}."
            )

        out = self.decoder(
            z_inst, z_sce,
            T=T,
            teacher_idx=teacher_idx,
            beam_train=beam_train,
            beam_size=beam_size,
            beam_expand=beam_expand,
            length_norm_alpha=length_norm_alpha,
            mix_teacher_p=mix_teacher_p,
            instance=instances,
            problem=problem,
        )

        if not isinstance(out, dict) or "logits_steps" not in out:
            raise RuntimeError("Stepwise decoder must return dict with key 'logits_steps'.")

        if return_aux:
            out = dict(out)
            out["z_inst"] = z_inst
            out["z_sce"] = z_sce
        return out
