# run_inference.py

import torch
import sys
import os
import json
import time

# ------------------------------------------------------------
# BYPASS SETUP (runtime quantization OFF)
# ------------------------------------------------------------
if os.getenv("GHOST_BYPASS", "True") == "True":
    import trion_core.quantization
    trion_core.quantization.quantize_158bit = lambda x: x

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from trion_core.modeling import GhostModel
from trion_core.config.model_config import GhostConfig
from trion_core.utils.sampling import (
    compute_logit_stats,
    adaptive_sample,
    GLOBAL_SAMPLING_STATS,  # 🔥 RUNTIME STATS
)

# ------------------------------------------------------------
# GENERATION (FINAL, ADAPTIVE, ALIAS-SAFE)
# ------------------------------------------------------------
def generate_sampled(
    model,
    input_ids,
    max_new_tokens=45,
    eos_token_id=50256,
):
    """
    Ghost / Trion Core – FINAL SAMPLING
    Adaptive: CUDA nucleus ↔ Python fallback
    """

    # 🔒 LOCKED SAMPLING PARAMS
    TEMPERATURE = 1.25
    TOP_K = 25
    TOP_P = 0.85
    REPETITION_PENALTY = 1.1

    generated = input_ids

    with torch.no_grad():
        for step in range(max_new_tokens):

            step_t0 = time.perf_counter()

            # ---------------------------------
            # FORWARD
            # ---------------------------------
            logits = model(generated)               # [B, T, V]
            next_logits = logits[:, -1, :].clone()  # 🔒 alias break

            # ---------------------------------
            # REPETITION PENALTY
            # ---------------------------------
            used_tokens = set(generated[0].tolist())
            for t in used_tokens:
                next_logits[0, t] /= REPETITION_PENALTY

            # ---------------------------------
            # LOGIT STATS (VOCAB SIDE)
            # ---------------------------------
            stats = compute_logit_stats(next_logits[0])
            print("LOGITS DEVICE:", next_logits.device)

            # ---------------------------------
            # OPTIONAL: HEAD ENTROPY (MODEL SIDE)
            # ---------------------------------
            head_info = ""
            if hasattr(model, "last_head_stats"):
                hs = model.last_head_stats
                head_info = (
                    f" | Hh={hs.get('head_entropy_mean', 0):.2f}"
                    f" σh={hs.get('head_entropy_std', 0):.2f}"
                    f" Ah={hs.get('active_heads', 0)}"
                )

            print(
                f"[{step:03d}] "
                f"Hv={stats['entropy']:.2f} | "
                f"T1={stats['top1_prob']:.2f} | "
                f"T5={stats['top5_prob']:.2f} | "
                f"Veff={stats['effective_vocab']:.0f}"
                f"{head_info}"
            )

            # ---------------------------------
            # 🔥 ADAPTIVE SAMPLING
            # ---------------------------------
            next_token_id = adaptive_sample(
                next_logits[0],
                stats=stats,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            )

            next_token = torch.tensor(
                [[next_token_id]],
                device=generated.device,
                dtype=torch.long,
            )

            generated = torch.cat([generated, next_token], dim=1)

            step_ms = (time.perf_counter() - step_t0) * 1000.0
            print(f"        step_time={step_ms:.2f} ms")

            # ---------------------------------
            # EOS
            # ---------------------------------
            if next_token_id == eos_token_id:
                break

    return generated


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = GhostConfig()

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        state = torch.load(cfg.brain_path, map_location=device)

        cfg.vocab_size = state["token_embedding.weight"].shape[0]
        cfg.d_model = state["token_embedding.weight"].shape[1]

        model = GhostModel(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        prompt = "The artificial intelligence system is designed to"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        GLOBAL_SAMPLING_STATS.reset()

        output_ids = generate_sampled(
            model,
            input_ids,
            max_new_tokens=45,
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ---------------------------------
        # 🔥 SAMPLING SUMMARY
        # ---------------------------------
        summary = GLOBAL_SAMPLING_STATS.summary()
        print("\n=== SAMPLING SUMMARY ===")
        for k, v in summary.items():
            print(
                f"{k:>16} | "
                f"ratio={v['ratio']:.2%} | "
                f"avg_sampling={v['avg_sampling_ms']:.3f} ms"
            )

        print(json.dumps({
            "status": "success",
            "text": text.replace("\n", " ").strip(),
            "sampling_summary": summary,
        }))

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }))


if __name__ == "__main__":
    main()
