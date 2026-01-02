import torch
import math
import time

# ------------------------------------------------------------
# OPTIONAL CUDA NUCLEUS EXTENSION (PACKAGE-AWARE)
# ------------------------------------------------------------
try:
    # Correct import path for packaged .pyd
    from trion_core.engine.nucleus import trion_nucleus
    HAS_CUDA_NUCLEUS = True
except Exception as _e:
    trion_nucleus = None
    HAS_CUDA_NUCLEUS = False


# ------------------------------------------------------------
# RUNTIME SAMPLING STATS (GLOBAL, LIGHTWEIGHT)
# ------------------------------------------------------------
class _SamplingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = {
            "cuda_nucleus": 0,
            "python_topk_topp": 0,
        }
        self.time_ms = {
            "cuda_nucleus": 0.0,
            "python_topk_topp": 0.0,
        }

    def log(self, path: str, sampling_time_ms: float):
        if path in self.count:
            self.count[path] += 1
            self.time_ms[path] += sampling_time_ms

    def summary(self):
        total = self.count["cuda_nucleus"] + self.count["python_topk_topp"]
        out = {}
        for k in self.count:
            c = self.count[k]
            out[k] = {
                "count": c,
                "ratio": c / max(total, 1),
                "avg_sampling_ms": self.time_ms[k] / max(c, 1),
            }
        return out


GLOBAL_SAMPLING_STATS = _SamplingStats()


# ------------------------------------------------------------
# LOGIT STATISTICS
# ------------------------------------------------------------
@torch.no_grad()
def compute_logit_stats(logits: torch.Tensor):
    """
    logits: [V]
    returns dict with entropy-driven metrics
    """

    finite_mask = torch.isfinite(logits)
    if not finite_mask.any():
        return {
            "entropy": 0.0,
            "top1_prob": 1.0,
            "top5_prob": 1.0,
            "effective_vocab": 1.0,
        }

    safe_logits = logits[finite_mask]

    log_probs = torch.log_softmax(safe_logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum().item()
    top1_prob = probs.max().item()

    k = min(5, probs.numel())
    top5_prob = torch.topk(probs, k=k).values.sum().item()

    effective_vocab = math.exp(entropy) if entropy > 0 else 1.0

    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "effective_vocab": effective_vocab,
    }


# ------------------------------------------------------------
# PYTHON TOP-K + TOP-P (ALIAS-SAFE)
# ------------------------------------------------------------
@torch.no_grad()
def sample_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.3,
    top_k: int = 40,
    top_p: float = 0.9,
):
    """
    logits: [V]
    returns: token_id (int)
    """

    # 🔒 alias break (CRITICAL)
    logits = logits.clone()
    V = logits.numel()

    if not torch.isfinite(logits).any():
        return torch.randint(0, V, (1,), device=logits.device).item()

    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature

    # -----------------------------
    # Top-K
    # -----------------------------
    if 0 < top_k < V:
        topk_vals, _ = torch.topk(logits, top_k)
        kth = topk_vals[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    # -----------------------------
    # Sort for nucleus
    # -----------------------------
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    nucleus_mask = cumulative_probs <= top_p
    nucleus_mask[0] = True

    sorted_logits = torch.where(
        nucleus_mask,
        sorted_logits,
        torch.full_like(sorted_logits, float("-inf")),
    )

    final_logits = torch.full_like(logits, float("-inf"))
    final_logits.scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(final_logits, dim=-1)

    if torch.isnan(probs).any() or probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.numel()

    return torch.multinomial(probs, 1).item()


# ------------------------------------------------------------
# 🔥 ADAPTIVE SAMPLER (CUDA NUCLEUS ↔ PYTHON)
# ------------------------------------------------------------
@torch.no_grad()
def adaptive_sample(
    logits: torch.Tensor,
    stats: dict,
    temperature: float,
    top_k: int,
    top_p: float,
):
    """
    Chooses CUDA nucleus or Python sampling based on entropy.
    Logs runtime usage and sampling cost.
    """

    Hv = float(stats.get("entropy", 0.0))
    Veff = float(stats.get("effective_vocab", 1.0))

    t0 = time.perf_counter()

    # -----------------------------------------
    # CUDA NUCLEUS (HIGH ENTROPY REGIME)
    # -----------------------------------------
    if (
        HAS_CUDA_NUCLEUS
        and logits.is_cuda
        and Hv >= 6.5
        and Veff >= 512
    ):
        token = trion_nucleus.nucleus_sample(
            logits,
            float(max(temperature, 1e-6)),
            float(top_p),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        GLOBAL_SAMPLING_STATS.log("cuda_nucleus", dt_ms)
        return int(token.item())

    # -----------------------------------------
    # PYTHON FALLBACK
    # -----------------------------------------
    token = sample_topk_topp(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    GLOBAL_SAMPLING_STATS.log("python_topk_topp", dt_ms)
    return token
