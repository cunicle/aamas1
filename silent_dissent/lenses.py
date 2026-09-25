"""Lenses: decode an intermediate residual into vocabulary logits.

Every lens implements two methods used by the rest of the pipeline:

    logits(h, layer)          h: [B, d] residual after decoder block `layer`
                              -> [B, V] vocabulary logits
    direction(token, layer)   -> [d] residual-space direction that increases the
                              lens logit of `token` at `layer` (used for
                              injection in the causal test)

`LogitLens` is fully implemented and serves as the baseline. `AffineLens`
covers any lens that is a learned/derived per-layer linear map into the final
residual space (tuned lens and similar). `JLens` is the hook for the J-lens
implementation used in the paper.
"""
from __future__ import annotations

import torch

from .model import get_final_norm


class Lens:
    name = "base"

    def logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def direction(self, token_id: int, layer: int) -> torch.Tensor:
        raise NotImplementedError


def _norm_gain(norm: torch.nn.Module) -> torch.Tensor | None:
    """Elementwise gain of the final norm (Gemma-style norms scale by 1 + w)."""
    w = getattr(norm, "weight", None)
    if w is None:
        return None
    if "gemma" in type(norm).__name__.lower():
        return 1.0 + w
    return w


class LogitLens(Lens):
    """unembed(final_norm(h)). At the last layer this reproduces the model logits."""

    name = "logit"

    def __init__(self, model):
        self.norm = get_final_norm(model)
        self.unembed = model.get_output_embeddings()

    def _decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.norm(h))

    def logits(self, h, layer):
        return self._decode(h)

    def direction(self, token_id, layer):
        w = self.unembed.weight[token_id].float()
        gain = _norm_gain(self.norm)
        return w * gain.float() if gain is not None else w


class AffineLens(LogitLens):
    """unembed(final_norm(A_l h + b_l)) with per-layer translators loaded from disk.

    File format (torch.save): {layer_index: {"weight": [d, d], "bias": [d]}}.
    Layers missing from the file fall back to the identity (plain logit lens).
    """

    name = "affine"

    def __init__(self, model, path: str):
        super().__init__(model)
        dev = self.unembed.weight.device
        raw = torch.load(path, map_location=dev)
        self.maps = {int(k): (v["weight"].to(dev), v.get("bias")) for k, v in raw.items()}

    def _translate(self, h, layer):
        if layer not in self.maps:
            return h
        W, b = self.maps[layer]
        out = h.to(W.dtype) @ W.T
        if b is not None:
            out = out + b.to(out.dtype)
        return out.to(h.dtype)

    def logits(self, h, layer):
        return self._decode(self._translate(h, layer))

    def direction(self, token_id, layer):
        d = super().direction(token_id, layer)
        if layer not in self.maps:
            return d
        return self.maps[layer][0].float().T @ d  # pull back through A_l


class JLens(Lens):
    """J-lens. TODO: implement `logits` and `direction`.

    If the J-lens is a per-layer linear map into the final residual space, the
    quickest route is to export it in `AffineLens` format and set
    `lens: {name: affine, path: ...}` in the config instead of filling this in.
    """

    name = "jlens"

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def logits(self, h, layer):
        raise NotImplementedError("JLens.logits: plug in the J-lens readout here")

    def direction(self, token_id, layer):
        raise NotImplementedError("JLens.direction: plug in the J-lens token direction here")


LENSES = {"logit": LogitLens, "affine": AffineLens, "jlens": JLens}


def build_lens(model, name: str = "logit", **kwargs) -> Lens:
    return LENSES[name](model, **kwargs)
