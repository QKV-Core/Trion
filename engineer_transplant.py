# engineer_transplant.py
import torch
import sys
import os
import io

# --------------------------------------------------
# SAFE STDOUT (Windows cp1252 fix)
# --------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import OPTForCausalLM
from trion_core.modeling import GhostModel
from trion_core.config.model_config import GhostConfig

# --------------------------------------------------
# HEAD-AWARE DENSITY SCHEDULE
# --------------------------------------------------
def head_density(base_keep: float, head_idx: int, n_head: int) -> float:
    if head_idx < int(n_head * 0.33):
        mult = 1.2
    elif head_idx < int(n_head * 0.66):
        mult = 1.0
    else:
        mult = 0.8

    return float(min(0.9, max(0.6, base_keep * mult)))

# --------------------------------------------------
# Q-ONLY TERNARY (HEAD-AWARE)
# --------------------------------------------------
def ternary_q_only(weight, base_keep, n_head, energy_fix=True):
    with torch.no_grad():
        d_model = weight.shape[0]
        head_dim = d_model // n_head
        out = torch.zeros_like(weight)

        for h in range(n_head):
            s = h * head_dim
            e = s + head_dim
            w_h = weight[s:e]

            keep = head_density(base_keep, h, n_head)

            abs_w = w_h.abs()
            k = max(1, int(abs_w.numel() * keep))
            thresh = torch.topk(abs_w.view(-1), k).values[-1]

            mask = abs_w >= thresh
            w_t = torch.sign(w_h) * mask

            if energy_fix:
                src_std = w_h.std()
                tgt_std = w_t.std().clamp(min=1e-8)
                ratio = (src_std / tgt_std).clamp(0.5, 3.0)
                w_t = w_t * ratio

            out[s:e] = w_t

        return out

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    BASE_QK_KEEP = float(os.getenv("GHOST_QK_KEEP", "0.6"))
    QK_MODE = os.getenv("GHOST_QK_MODE", "ternary")
    ENERGY_FIX = os.getenv("GHOST_ENERGY_FIX", "True") == "True"

    print(f"[V15] Q-ONLY TERNARY | ENV base={BASE_QK_KEEP}")
    print(f"[DEBUG] ENV GHOST_QK_KEEP = {os.getenv('GHOST_QK_KEEP')}")

    if QK_MODE != "ternary":
        raise RuntimeError("Only Q-only ternary supported")

    device = "cpu"

    donor = OPTForCausalLM.from_pretrained("facebook/opt-125m").to(device).eval()
    dsd = donor.state_dict()
    dcfg = donor.config

    cfg = GhostConfig(
        vocab_size=dcfg.vocab_size,
        d_model=dcfg.hidden_size,
        n_layer=dcfg.num_hidden_layers,
        n_head=dcfg.num_attention_heads,
        max_seq_len=dcfg.max_position_embeddings,
    )

    ghost = GhostModel(cfg).to(device)
    gsd = ghost.state_dict()

    # Embeddings
    gsd["token_embedding.weight"] = dsd["model.decoder.embed_tokens.weight"]
    gsd["position_embedding.weight"] = dsd["model.decoder.embed_positions.weight"]

    for i in range(cfg.n_layer):
        g = f"layers.{i}"
        d = f"model.decoder.layers.{i}"

        gsd[f"{g}.attn.q_proj.weight"] = ternary_q_only(
            dsd[f"{d}.self_attn.q_proj.weight"],
            BASE_QK_KEEP,
            cfg.n_head,
            ENERGY_FIX,
        )
        gsd[f"{g}.attn.q_proj.bias"] = dsd[f"{d}.self_attn.q_proj.bias"]

        gsd[f"{g}.attn.k_proj.weight"] = dsd[f"{d}.self_attn.k_proj.weight"]
        gsd[f"{g}.attn.k_proj.bias"]   = dsd[f"{d}.self_attn.k_proj.bias"]

        gsd[f"{g}.attn.v_proj.weight"] = dsd[f"{d}.self_attn.v_proj.weight"]
        gsd[f"{g}.attn.v_proj.bias"]   = dsd[f"{d}.self_attn.v_proj.bias"]

        gsd[f"{g}.attn.o_proj.weight"] = dsd[f"{d}.self_attn.out_proj.weight"]
        gsd[f"{g}.attn.o_proj.bias"]   = dsd[f"{d}.self_attn.out_proj.bias"]

        gsd[f"{g}.mlp.fc1.weight"] = dsd[f"{d}.fc1.weight"]
        gsd[f"{g}.mlp.fc1.bias"]   = dsd[f"{d}.fc1.bias"]
        gsd[f"{g}.mlp.fc2.weight"] = dsd[f"{d}.fc2.weight"]
        gsd[f"{g}.mlp.fc2.bias"]   = dsd[f"{d}.fc2.bias"]

        gsd[f"{g}.ln1.weight"] = dsd[f"{d}.self_attn_layer_norm.weight"]
        gsd[f"{g}.ln1.bias"]   = dsd[f"{d}.self_attn_layer_norm.bias"]
        gsd[f"{g}.ln2.weight"] = dsd[f"{d}.final_layer_norm.weight"]
        gsd[f"{g}.ln2.bias"]   = dsd[f"{d}.final_layer_norm.bias"]

    gsd["norm.weight"] = dsd["model.decoder.final_layer_norm.weight"]
    gsd["norm.bias"]   = dsd["model.decoder.final_layer_norm.bias"]
    gsd["output_head.weight"] = dsd["lm_head.weight"]

    ghost.load_state_dict(gsd, strict=True)
    torch.save(ghost.state_dict(), "trion_brain_opt_ghost.pt")

    print("OK – Head-aware Q-only ternary brain written.")

if __name__ == "__main__":
    main()
