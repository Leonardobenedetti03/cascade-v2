"""Compute cost (GMACs) of the cascade.

Cost model per sample:

- CamFormer (stage 1 of the cascade) and the gate **always** run;
- TimeSformer-EgoVLP (stage 2) runs only on the escalated samples.

So the mean cost per sample, with an escalation share ``f``, is::

    gmacs = gmacs_s1 + gate_gmacs + f * gmacs_s2

All functions are pure (numpy, no I/O).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

# Relative tolerance for the comparisons with the budget (so that 0.1*3 != 0.3 does not drop an escalation).
_REL_TOL = 1e-9


def _check_non_negative(**costs: float) -> None:
    for name, value in costs.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} deve essere finito e >= 0, ricevuto {value!r}")


def macs_to_gmacs(macs: float) -> float:
    """Convert a number of MACs to GMACs (1 GMAC = 1e9 MAC)."""
    _check_non_negative(macs=macs)
    return float(macs) / 1e9


def cascade_gmacs(
    escalated,
    gmacs_s1: float,
    gmacs_s2: float,
    gate_gmacs: float = 0.0,
) -> float:
    """Mean cost per sample of the cascade, in GMACs.

    Args:
        escalated: 1D boolean (or 0/1) array, True if the sample goes to stage 2.
        gmacs_s1: cost per sample of CamFormer.
        gmacs_s2: cost per sample of TimeSformer-EgoVLP.
        gate_gmacs: cost per sample of the gate.
    """
    _check_non_negative(gmacs_s1=gmacs_s1, gmacs_s2=gmacs_s2, gate_gmacs=gate_gmacs)
    esc = np.asarray(escalated)
    if esc.ndim != 1 or esc.size == 0:
        raise ValueError("escalated deve essere un array 1D non vuoto")
    if not np.isin(esc, (0, 1)).all():
        raise ValueError("escalated deve contenere solo valori booleani/0-1")
    frac = float(esc.astype(bool).mean())
    return gmacs_s1 + gate_gmacs + frac * gmacs_s2


def max_escalations(
    n: int,
    budget_gmacs: float,
    gmacs_s1: float,
    gmacs_s2: float,
    gate_gmacs: float = 0.0,
) -> int:
    """Largest number of samples (out of ``n``) we can escalate while staying within the mean budget.

    Raises:
        ValueError: if the budget is below the minimum cost (stage 1 + gate), which is paid on every
            sample anyway.
    """
    _check_non_negative(gmacs_s1=gmacs_s1, gmacs_s2=gmacs_s2, gate_gmacs=gate_gmacs)
    if n <= 0:
        raise ValueError(f"n deve essere > 0, ricevuto {n}")
    base = gmacs_s1 + gate_gmacs
    tol = _REL_TOL * max(1.0, abs(budget_gmacs))
    if budget_gmacs < base - tol:
        raise ValueError(f"budget {budget_gmacs} GMACs sotto il costo minimo {base} GMACs (stadio 1 + gate)")
    if gmacs_s2 == 0:
        return n
    k = math.floor(n * (budget_gmacs - base) / gmacs_s2 + _REL_TOL * n)
    return int(min(max(k, 0), n))


def relative_cost(gmacs: float, gmacs_reference: float) -> float:
    """Cost relative to a reference (e.g. TimeSformer alone): ``gmacs / gmacs_reference``."""
    _check_non_negative(gmacs=gmacs, gmacs_reference=gmacs_reference)
    if gmacs_reference == 0:
        raise ValueError("gmacs_reference deve essere > 0")
    return gmacs / gmacs_reference


# Measuring the cost (stage 40)
def camformer_macs(T: int, arch, n_classes: int) -> int:
    """Analytic MACs of one CamFormer forward pass (batch 1, ``T`` poses, ``pooling='no'``).

    From the code in ``seeing_without_pixels/models/cm_encoder.py``: pose projection (9 dims with ``rel9d_grav``,
    plus the gravity token projected separately), ``num_layers`` transformer layers on ``n = T (+1)`` tokens (QKV,
    attention scores and weighted sum, output projection, FFN), ``out_proj`` on all T tokens, head.
    Norms and softmax are left out (as in the counters).
    """
    d, L, ff, D = arch.d_model, arch.num_layers, arch.dim_feedforward, arch.output_dim
    grav = 1 if arch.pose_encoding == "rel9d_grav" else 0
    pose_dim = {"abs7d": 7, "rel9d": 9, "rel9d_grav": 9}[arch.pose_encoding]
    n = T + grav
    in_proj = T * pose_dim * d + grav * 3 * d
    layer = n * 3 * d * d + 2 * n * n * d + n * d * d + 2 * n * d * ff
    out = (T * d * D if D != d else 0) + D * n_classes
    return int(in_proj + L * layer + out)


def timesformer_macs(arch, num_frames: int) -> int:
    """Analytic MACs of the EgoVLP video branch (SpaceTimeTransformer + vid_proj), batch 1.

    From ``EgoVLP/model/video_transformer.py``: patch embedding; for each block a temporal attention (groups =
    patches per frame, F tokens + CLS) and a spatial one (groups = frames, P tokens + CLS), each with QKV, the CLS
    attending to all N tokens, the groups and the projection; MLP with ratio 4.
    """
    d, p, img, depth = arch.embed_dim, arch.patch_size, arch.img_size, arch.depth
    P = (img // p) ** 2
    F = num_frames
    N = 1 + F * P

    def var_attn(groups: int, size: int) -> int:
        return N * 3 * d * d + 2 * N * d + 2 * groups * size * (size + 1) * d + N * d * d

    block = var_attn(P, F) + var_attn(F, P) + 2 * N * d * 4 * d
    return int(F * P * 3 * p * p * d + depth * block + d * arch.proj_dim)


def _sdpa_macs(q_shape, k_shape, v_shape) -> int:
    batch = 1
    for x in q_shape[:-2]:
        batch *= int(x)
    L, E = int(q_shape[-2]), int(q_shape[-1])
    S, Ev = int(k_shape[-2]), int(v_shape[-1])
    return batch * L * S * (E + Ev)


def _count_torch(model, inputs: tuple) -> float:
    """MACs from ``torch.utils.flop_counter`` (FLOPs / 2), with the SDPA variants mapped by hand: some torch
    versions (e.g. 2.8 on CPU) do not count them and underestimate the attention."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    def sdpa_flops(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
        return 2 * _sdpa_macs(query_shape, key_shape, value_shape)

    mapping = {}
    for name in (
        "_scaled_dot_product_flash_attention_for_cpu",
        "_scaled_dot_product_attention_math",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
    ):
        op = getattr(torch.ops.aten, name, None)
        if op is not None:
            mapping[op] = sdpa_flops
    try:
        mode = FlopCounterMode(display=False, custom_mapping=mapping)
    except TypeError:  # torch without custom_mapping
        mode = FlopCounterMode(display=False)
    with torch.no_grad(), mode as fc:
        model(*inputs)
    return fc.get_total_flops() / 2.0


def _count_fvcore(model, inputs: tuple):
    """MACs from fvcore (one op per multiply-add), with a handle for SDPA: fvcore does not count it, and
    CamFormer (nn.MultiheadAttention without the fast path) goes through it."""
    import torch
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape

    def sdpa_handle(inputs_, outputs_):
        return _sdpa_macs(get_shape(inputs_[0]), get_shape(inputs_[1]), get_shape(inputs_[2]))

    with torch.no_grad():
        fca = FlopCountAnalysis(model, inputs)
        fca.set_op_handle("aten::scaled_dot_product_attention", sdpa_handle)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total()), {k: int(v) for k, v in fca.unsupported_ops().items()}


def measure_gmacs(
    model, inputs: tuple, counter: str = "fvcore", analytic_macs: Optional[int] = None, tol: float = 0.02
) -> dict:
    """GMACs of one forward pass at batch 1: fvcore, ``torch.utils.flop_counter`` and the analytic formula.

    The fast path of ``nn.TransformerEncoderLayer`` (CamFormer in eval mode) hides almost every op from the
    counters, so we turn it off while counting. ``gmacs`` is the value from ``counter`` (``fvcore`` | ``torch`` |
    ``analytic``), with a fallback if that counter fails. If a counter is more than ``tol`` away from the
    analytic formula, we say so in the log.
    """
    import logging

    import torch

    log = logging.getLogger(__name__)
    model.eval()
    prev = None
    try:
        prev = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:  # old torch: no setter
        pass
    out: Dict[str, object] = {"gmacs_analytic": macs_to_gmacs(analytic_macs) if analytic_macs else None}
    try:
        out["gmacs_torch"] = macs_to_gmacs(_count_torch(model, inputs))
        try:
            total, unsupported = _count_fvcore(model, inputs)
            out.update({"gmacs_fvcore": macs_to_gmacs(total), "unsupported_ops": unsupported})
        except Exception as e:  # noqa: BLE001 - fvcore cannot trace every model
            log.warning("fvcore non riesce a contare il modello (%s: %s)", type(e).__name__, e)
            out.update({"gmacs_fvcore": None, "fvcore_error": f"{type(e).__name__}: {e}"})
    finally:
        if prev is not None:
            torch.backends.mha.set_fastpath_enabled(prev)
    order = [counter] + [c for c in ("fvcore", "analytic", "torch") if c != counter]
    use = next(c for c in order if out.get(f"gmacs_{c}") is not None)
    out["gmacs"], out["method"] = out[f"gmacs_{use}"], use
    ref = out["gmacs_analytic"]
    if ref:
        for c in ("fvcore", "torch"):
            v = out.get(f"gmacs_{c}")
            if v is not None and abs(v / ref - 1) > tol:
                log.warning(
                    "contatore %s: %.5g GMACs, formula analitica %.5g (scarto %.1f%%)", c, v, ref, 100 * (v / ref - 1)
                )
    return out


def measure_latency_ms(model, inputs: tuple, reps: int = 20, warmup: int = 3) -> dict:
    import time

    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {"latency_ms": 1000.0 * (time.perf_counter() - t0) / max(reps, 1), "device": str(device)}
