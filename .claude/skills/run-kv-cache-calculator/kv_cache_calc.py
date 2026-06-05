#!/usr/bin/env python3
"""KV-cache offloading-crossover simulator.

Estimates the concurrency at which the device KV pool overflows — where
hicache (host RAM) / lmcache offloading starts to pay off — for a model on
given hardware + parallelism, and prints a per-layout table.

Model params come from one of:
  --config-json  vendor/modelconfig.json (offline, default) or any JSON
                 in the same {model: {hidden_size, ...}} shape
  --hf           live transformers.AutoConfig.from_pretrained(model)
  --hidden-size/--layers/... explicit overrides (no lookup needed)
"""
import argparse
import json
import os
import re
import sys

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "fp8": 1}

# Bytes per parameter by quantization of the *weights* (not the KV cache).
# 4-bit formats are >0.5: the packed weight is ~4.25 bits (4-bit mantissa + an
# 8-bit block scale per 32 elems), and embeddings/norms/router/attention stay in
# higher precision. Calibrated to 0.57 against the measured device KV pool of the
# GLM-5.1-MXFP4 mi355x run (startup_available 41GB -> weight ~199GB/gpu @tp2,
# util 0.85 -> 0.576 B/param); without this the pool is ~2x over-counted.
QUANT_BYTES = {"fp4": 0.57, "mxfp4": 0.57, "int4": 0.57, "fp8": 1.0,
               "int8": 1.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0,
               "float16": 2.0}

GIB = 1024 ** 3

# Per-GPU HBM (GB) by hardware. Mirrors vendor/aliases.json hw_hbm_gb.
HW_HBM_GB = {"mi300x": 192, "mi325x": 256, "mi355x": 288,
             "h100": 80, "h200": 141, "b200": 192, "b300": 288}

# Activation / framework reserve carved out of the mem-fraction budget before KV
# (CUDA graphs, comms buffers, intermediate activations). Rough, per GPU.
ACT_RESERVE_GB = 8.0


def _is_deepseek(model):
    return model in ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1")


def _is_gqa_headdim(model):
    m = model.lower()
    return "qwen/qwen3-" in m or model.startswith("zai-org/GLM-4.")


def kv_formula(model, c):
    """Pick the KV formula. Prefer config contents over model-id matching so
    explicit-override and unknown models (newer than the id heuristics) still
    get the right math.
      mla      -> latent KV: layers * (kv_lora_rank + qk_rope_head_dim)
      head_dim -> GQA with explicit head_dim: 2 * layers * kv_heads * head_dim
      derived  -> standard: 2 * layers * kv_heads * (hidden_size/attn_heads)
    """
    if c.get("kv_lora_rank") and c.get("qk_rope_head_dim"):
        return "mla"
    if _is_deepseek(model):
        return "mla"
    if c.get("head_dim"):
        return "head_dim"
    if _is_gqa_headdim(model):
        return "head_dim"
    return "derived"


def elements_per_token(model, c):
    """KV elements per token (summed over all layers)."""
    layers = c["num_hidden_layers"]
    f = kv_formula(model, c)
    if f == "mla":
        # MLA: latent kv_lora_rank + rope dim, single (not doubled) per layer.
        per_layer = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        # DSA (GlmMoeDsa / DeepSeek-V3.2+): the lightning indexer keeps its own
        # tiny per-token key cache — MQA, one shared key of index_head_dim per
        # layer, no value (it only scores). vLLM: ~132 B/token vs 576 for MLA.
        if c.get("index_head_dim"):
            per_layer += c["index_head_dim"]
        return layers * per_layer
    if f == "head_dim":
        return 2 * layers * c["num_key_value_heads"] * c["head_dim"]
    head_size = c["hidden_size"] / c["num_attention_heads"]
    return 2 * layers * c["num_key_value_heads"] * head_size


def load_config(model, args):
    if args.hidden_size or args.layers:
        cfg = {
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.attn_heads,
            "num_hidden_layers": args.layers,
            "num_key_value_heads": args.kv_heads or args.attn_heads,
        }
        if args.head_dim:
            cfg["head_dim"] = args.head_dim
        if args.kv_lora_rank:
            cfg["kv_lora_rank"] = args.kv_lora_rank
            cfg["qk_rope_head_dim"] = args.qk_rope_head_dim
        return cfg

    if args.hf:
        from transformers import AutoConfig
        ac = AutoConfig.from_pretrained(model)
        cfg = {
            "hidden_size": getattr(ac, "hidden_size", None),
            "num_attention_heads": getattr(ac, "num_attention_heads", None),
            "num_hidden_layers": getattr(ac, "num_hidden_layers", None),
            "num_key_value_heads": getattr(ac, "num_key_value_heads", None),
        }
        hd = getattr(ac, "head_dim", None)
        if hd:
            cfg["head_dim"] = hd
        if _is_deepseek(model):
            cfg["kv_lora_rank"] = getattr(ac, "kv_lora_rank", None)
            cfg["qk_rope_head_dim"] = getattr(ac, "qk_rope_head_dim", None)
        return cfg

    with open(args.config_json) as fh:
        configs = json.load(fh)
    if model not in configs:
        sys.exit(
            f"error: model '{model}' not in {args.config_json}. "
            f"Use --hf for a live lookup, or --list to see known models."
        )
    return configs[model]


def estimate_params(model, c):
    """Estimate total resident parameter count from config fields.

    For MoE models every expert is resident in HBM (vLLM/sglang load them all),
    so "total" counts all experts, not just the active/hot ones. Returns total
    param count (float). Best-effort: good to ~10% for the families we ship.
    """
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    vocab = c.get("vocab_size", 0)

    # --- attention block params per layer ---
    if kv_formula(model, c) == "mla":
        # MLA: q/kv down-projections (LoRA) + up-projections + o_proj. Use the
        # explicit nope/rope/v dims when present, else fall back to head_dim.
        n_heads = c["num_attention_heads"]
        qk_nope = c.get("qk_nope_head_dim", c.get("head_dim", 128))
        qk_rope = c.get("qk_rope_head_dim", 64)
        v_dim = c.get("v_head_dim", c.get("head_dim", 128))
        kv_lora = c.get("kv_lora_rank", 512)
        q_lora = c.get("q_lora_rank") or 0
        qk_head = qk_nope + qk_rope
        if q_lora:
            q_params = h * q_lora + q_lora * n_heads * qk_head
        else:
            q_params = h * n_heads * qk_head
        kv_down = h * (kv_lora + qk_rope)
        kv_up = kv_lora * n_heads * (qk_nope + v_dim)
        o_proj = n_heads * v_dim * h
        attn = q_params + kv_down + kv_up + o_proj
    else:
        n_heads = c["num_attention_heads"]
        kv_heads = c["num_key_value_heads"]
        hd = c.get("head_dim", h // n_heads)
        q = h * n_heads * hd
        kv = 2 * h * kv_heads * hd
        o = n_heads * hd * h
        attn = q + kv + o

    # --- MLP params per layer (dense vs MoE) ---
    n_experts = (c.get("num_experts") or c.get("n_routed_experts")
                 or c.get("num_local_experts") or 0)
    inter = c.get("intermediate_size", 0)
    moe_inter = c.get("moe_intermediate_size", 0)
    n_shared = c.get("n_shared_experts", 0) or 0
    first_dense = c.get("first_k_dense_replace", 0) or 0

    def dense_mlp(d):
        return 3 * h * d  # gate + up + down

    if n_experts:
        # gpt-oss / MiniMax expose only intermediate_size for experts; DeepSeek/
        # GLM/Qwen expose moe_intermediate_size. Prefer the MoE-specific one.
        e_inter = moe_inter or inter
        routed = n_experts * dense_mlp(e_inter)
        shared = n_shared * dense_mlp(moe_inter or inter) if n_shared else 0
        moe_layer = routed + shared
        dense_layer = dense_mlp(inter or e_inter)
        mlp_total = first_dense * dense_layer + (L - first_dense) * moe_layer
    else:
        mlp_total = L * dense_mlp(inter)

    attn_total = L * attn
    embed = 2 * vocab * h  # input embedding + output head (often untied)
    return attn_total + mlp_total + embed


def weight_gb(model, c, quant):
    bytes_per = QUANT_BYTES.get(quant, 1.0)
    return estimate_params(model, c) * bytes_per / GIB


def kv_shard_factor(model, c, layout, tp, dp):
    """How many GPUs the per-token KV of ONE sequence is split across.

    GQA/MHA: KV heads shard across the attention-TP group, capped at kv_heads
    (beyond that KV is replicated, no further split).
    MLA: the latent KV is not head-shardable -> never splits under TP (factor 1);
    a sequence's KV lives whole on its attention rank.
    DEP (dp-attention): attention TP=1, so no intra-sequence split either way.
    """
    f = kv_formula(model, c)
    if layout == "dep":
        return 1
    if f == "mla":
        return 1
    kv_heads = c.get("num_key_value_heads", 1)
    return max(1, min(kv_heads, tp))


def kv_replicas(model, c, layout, tp, dp):
    """How many INDEPENDENT KV pools exist across the node (each holds distinct
    sequences). pure-TP / TP+EP = one engine = 1 pool. DEP = dp independent
    attention ranks = dp pools."""
    if layout == "dep":
        return dp
    return 1


def _box_table(headers, rows):
    """Render a Unicode box-drawing table (header + per-row separators)."""
    cols = len(headers)
    widths = [len(str(headers[i])) for i in range(cols)]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i])))

    def fmt(cells):
        return "│ " + " │ ".join(str(c).ljust(widths[i])
                                 for i, c in enumerate(cells)) + " │"

    def sep(left, mid, right):
        return left + mid.join("─" * (widths[i] + 2) for i in range(cols)) + right

    out = [sep("┌", "┬", "┐"), fmt(headers), sep("├", "┼", "┤")]
    for j, row in enumerate(rows):
        out.append(fmt(row))
        if j < len(rows) - 1:
            out.append(sep("├", "┼", "┤"))
    out.append(sep("└", "┴", "┘"))
    return "\n".join(out)


def find_repo_root(start):
    """Walk up from `start` to the dir containing .github/configs."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, ".github", "configs")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def configured_layouts(model_id, hw, here, quant=None):
    """Find the parallelism layouts actually configured for (model, hw) in
    .github/configs/amd-master.yaml. Returns (specs, entry_names) where each
    spec is {layout, tp, dp, ep, ngpu, conc_list}, deduped. Empty if no match
    or pyyaml unavailable. If `quant` is given, only entries whose key/model
    advertise that weight quant are matched (fp4 and fp8 configs are distinct
    parallelism shapes, so mixing them is wrong)."""
    root = find_repo_root(here)
    if not root:
        return [], []
    cfg_path = os.path.join(root, ".github", "configs", "amd-master.yaml")
    try:
        import yaml
        with open(cfg_path) as fh:
            doc = yaml.safe_load(fh)
    except Exception:
        return [], []
    if not isinstance(doc, dict):
        return [], []

    # Match entries whose model id and runner match. The config keys aren't the
    # HF id, so match on the entry's `model:`/`runner:` fields, falling back to
    # the key substring.
    short = model_id.split("/")[-1].lower()
    specs, names, seen = [], [], set()
    for key, entry in doc.items():
        if not isinstance(entry, dict):
            continue
        runner = str(entry.get("runner", "")).lower()
        emodel = str(entry.get("model", "")).lower()
        if hw and runner and runner != hw:
            continue
        # Heuristic model match: HF short name shares a token with the entry
        # model id or the key (e.g. Kimi-K2.5 <-> amd/Kimi-K2.5-MXFP4).
        kmatch = short in emodel or short in key.lower() or \
            short.replace("-", "").replace(".", "") in \
            emodel.replace("-", "").replace(".", "")
        if not kmatch:
            continue
        # fp4 and fp8 configs are different parallelism shapes; only pull the
        # layouts that match the requested weight quant. 4-bit variants
        # (fp4/mxfp4/int4) are treated as one family.
        if quant:
            hay = (key.lower() + " " + emodel)
            q4 = quant in ("fp4", "mxfp4", "int4")
            want = ("fp4", "mxfp4", "int4") if q4 else (quant,)
            has_fp8 = "fp8" in hay
            has_4bit = any(t in hay for t in ("fp4", "mxfp4", "int4"))
            if has_fp8 or has_4bit:  # entry advertises a quant -> must match
                if not any(t in hay for t in want):
                    continue
        scen = entry.get("scenarios", {}) or {}
        for _stype, runs in scen.items():
            if not isinstance(runs, list):
                continue
            for run in runs:
                for ss in (run.get("search-space", []) or []):
                    tp = int(ss.get("tp", 1))
                    ep = int(ss.get("ep", 1))
                    dpattn = bool(ss.get("dp-attn", False))
                    if dpattn:
                        layout, dp, ngpu = "dep", tp, tp
                    elif ep > 1:
                        layout, dp, ngpu = "tp-ep", 1, tp
                    else:
                        layout, dp, ngpu = "tp", 1, tp
                    conc = ss.get("conc-list")
                    sig = (layout, tp, ep, dp)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    specs.append({"layout": layout, "tp": tp, "dp": dp,
                                  "ep": ep, "ngpu": ngpu, "conc_list": conc})
                    if key not in names:
                        names.append(key)
    return specs, names


def _row(model, c, dtype, w_gb, hbm, gpu_util, mean_isl,
         layout, tp, dp, ep, ngpu, conc_list=None):
    """Compute one sweep row for a concrete (layout, tp, dp, ep, ngpu) spec."""
    per_tok_total = elements_per_token(model, c) * DTYPE_BYTES[dtype]
    attn_tp = 1 if layout == "dep" else tp
    w_per_gpu = w_gb / ngpu
    pool_per_gpu = max(0.0, hbm * gpu_util - w_per_gpu - ACT_RESERVE_GB)
    shard = kv_shard_factor(model, c, layout, attn_tp, dp)
    replicas = kv_replicas(model, c, layout, attn_tp, dp)
    tokens_per_pool = pool_per_gpu * GIB * shard / per_tok_total \
        if per_tok_total else 0
    agg_tokens = tokens_per_pool * replicas
    crossover = agg_tokens / mean_isl if mean_isl else 0
    row = {
        "layout": layout, "tp": attn_tp, "dp": (dp if layout == "dep" else 1),
        "ep": ep, "ngpu": ngpu,
        "weight_gb_per_gpu": round(w_per_gpu, 1),
        "kv_pool_gb_per_gpu": round(pool_per_gpu, 1),
        # Effective aggregate KV capacity = the unique KV the node can hold before
        # overflow. GQA shards across nGPU (pools add up); MLA replicates the
        # latent KV on every TP rank, so only DEP's `replicas` scale it — adding
        # TP buys weight room, NOT KV room. This is the quantity the crossover
        # divides by ISL, so KV-cache / per-token-KV / ISL == crossover for all
        # layouts. (Physical KV HBM spent = per_gpu x ngpu, larger for MLA-TP.)
        "kv_pool_gb_total": round(pool_per_gpu * shard * replicas, 1),
        "kv_shard_factor": shard, "kv_pools": replicas,
        "node_kv_tokens": int(agg_tokens),
        "crossover_conc": round(crossover, 1),
        "infeasible": pool_per_gpu <= 0,
    }
    if conc_list is not None:
        row["conc_list"] = conc_list
    return row


def simulate(model, c, dtype, quant, hw, gpu_util, mean_isl,
             layouts=None, tp=4, dp=4, ep=8, specs=None):
    """Run the crossover sweep. If `specs` (a list of layout dicts from the
    batch config) is given, use those exact configured layouts; otherwise sweep
    the generic `layouts` at the given tp/dp/ep."""
    hbm = HW_HBM_GB.get(hw)
    per_tok_total = elements_per_token(model, c) * DTYPE_BYTES[dtype]
    w_gb = weight_gb(model, c, quant)
    params = estimate_params(model, c)

    rows = []
    if specs:
        for s in specs:
            if hbm is None:
                rows.append({"layout": s.get("layout", "?"), "error": "no --hw / hbm"})
                continue
            rows.append(_row(model, c, dtype, w_gb, hbm, gpu_util, mean_isl,
                             s["layout"], s["tp"], s["dp"], s["ep"], s["ngpu"],
                             s.get("conc_list")))
    else:
        for layout in (layouts or ["tp", "tp-ep", "dep"]):
            if hbm is None:
                rows.append({"layout": layout, "error": "no --hw / hbm"})
                continue
            ngpu = dp if layout == "dep" else tp
            ep_eff = 1 if layout == "tp" else ep
            rows.append(_row(model, c, dtype, w_gb, hbm, gpu_util, mean_isl,
                             layout, tp, dp, ep_eff, ngpu))

    return {
        "model": model, "hw": hw, "hbm_gb": hbm, "gpu_util": gpu_util,
        "dtype_kv": dtype, "quant_weights": quant,
        "params_total_b": round(params / 1e9, 1),
        "weight_gb_node": round(w_gb, 1),
        "per_token_kv_bytes": int(per_tok_total),
        "mean_isl": mean_isl, "rows": rows,
    }


def resolve_alias(token, here):
    """Resolve a CLI alias or launcher path/stem to (model_id, quant, hw).
    Returns (model_id, quant_or_None, hw_or_None) or (token, None, None) if it's
    already a bare model id / not resolvable."""
    apath = os.path.join(here, "vendor", "aliases.json")
    try:
        with open(apath) as fh:
            A = json.load(fh)
    except Exception:
        A = {"families": {}, "aliases": {}, "hw_hbm_gb": {}}

    # Launcher file path or stem -> parse family_precision_hw.
    stem = token
    if token.endswith(".sh") or os.sep in token:
        stem = os.path.basename(token)
        if stem.endswith(".sh"):
            stem = stem[:-3]
    parts = stem.split("_")
    if parts and parts[0] in A.get("families", {}):
        model_id = A["families"][parts[0]]
        quant = next((p for p in parts if p in QUANT_BYTES), None)
        hw = next((p for p in parts if p in HW_HBM_GB), None)
        # If a quant-specific alias exists for this family (e.g. the launcher
        # stem `kimik2.5_fp4` and alias `kimik2.5-fp4` -> amd/Kimi-K2.5-MXFP4),
        # prefer its canonical id so the .sh path agrees with the alias path.
        akey = f"{parts[0]}-{quant}" if quant else None
        if akey and akey in A.get("aliases", {}):
            a = A["aliases"][akey]
            return a["id"], a.get("quant", quant), hw
        return model_id, quant, hw

    # Direct alias.
    if token in A.get("aliases", {}):
        a = A["aliases"][token]
        return a["id"], a.get("quant"), a.get("hw")

    return token, None, None


def parse_launcher(path):
    """Pull server-side knobs from an agentic launcher .sh: kv-cache-dtype,
    gpu-memory-utilization / mem-fraction-static, max-model-len default."""
    out = {}
    try:
        with open(path) as fh:
            txt = fh.read()
    except Exception:
        return out
    m = re.search(r"--kv-cache-dtype[ =]+([^\s\\]+)", txt)
    if m:
        out["kv_dtype"] = "fp8" if m.group(1).startswith("fp8") else m.group(1)
    m = re.search(r"--gpu-memory-utilization[ =]+([0-9.]+)", txt) \
        or re.search(r"--mem-fraction-static[ =]+([0-9.]+)", txt)
    if m:
        out["gpu_util"] = float(m.group(1))
    m = re.search(r"MAX_MODEL_LEN=([0-9]+)", txt)
    if m and m.group(1) != "0":
        out["max_model_len"] = int(m.group(1))
    # Mean input length of the agentic trace the launcher replays. Measured
    # ~85k across ALL runs (CI agg_bmk.json: 45k–112k), independent of the
    # `..._256k` WEKA override — the "256k" is the trace's CONTEXT WINDOW /
    # max_model_len, NOT the mean prompt length. Validated: Kimi tp4 at 85k →
    # crossover ~41 matches the GPU-cache-hit collapse at conc 40 in CI. Crossover
    # scales inversely with this, so don't substitute max_model_len here (that
    # gives vllm's worst-case "max concurrency", a different question).
    out["mean_isl"] = 85000
    return out


def find_launcher(model_id, quant, hw, here, prefer=None):
    """Locate the agentic launcher .sh for (model_id, quant, hw) so an alias
    invocation can inherit the run's real util + trace ISL. 4-bit quants
    (fp4/mxfp4/int4) are one family. If several match (e.g. a vllm and an sglang
    variant), prefer one whose framework token appears in `prefer` (the configured
    layout names). Returns a path or None."""
    root = find_repo_root(here)
    if not root:
        return None
    ldir = os.path.join(root, "upstream", "InferenceX", "benchmarks",
                        "single_node", "agentic")
    if not os.path.isdir(ldir):
        return None
    def fam(mid):
        # Normalize a model id to a quant-agnostic family key so the alias id
        # (amd/GLM-5.1-MXFP4) matches the launcher-stem id (zai-org/GLM-5.1-FP8).
        s = mid.split("/")[-1].lower()
        for t in ("mxfp4", "fp4", "int4", "fp8", "int8", "bf16", "bfloat16",
                  "fp16"):
            s = s.replace(t, "")
        return re.sub(r"[^a-z0-9]", "", s)

    q4 = quant in ("fp4", "mxfp4", "int4") if quant else False
    target_fam = fam(model_id)
    cands = []
    for fn in sorted(os.listdir(ldir)):
        if not fn.endswith(".sh"):
            continue
        mid, lq, lhw = resolve_alias(fn[:-3], here)
        if fam(mid) != target_fam or (hw and lhw != hw):
            continue
        if quant:
            lq4 = lq in ("fp4", "mxfp4", "int4") if lq else False
            if (q4 and not lq4) or (not q4 and lq != quant):
                continue
        cands.append(os.path.join(ldir, fn))
    if not cands:
        return None
    if prefer:
        hay = " ".join(prefer).lower()
        for fw in ("sglang", "vllm", "trtllm"):
            if fw in hay:
                pref = [c for c in cands if fw in os.path.basename(c).lower()]
                if pref:
                    return pref[0]
    return cands[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?",
                   help="model id, alias (glm-5.1-fp8), or launcher .sh path")
    p.add_argument("--dtype", default=None, choices=list(DTYPE_BYTES),
                   help="KV cache dtype (default: from launcher, else fp8)")
    p.add_argument("--config-json", default=os.path.join(here, "vendor", "modelconfig.json"))
    p.add_argument("--hf", action="store_true", help="live transformers.AutoConfig lookup")
    p.add_argument("--list", action="store_true", help="list models in --config-json and exit")
    p.add_argument("--json", action="store_true", help="emit JSON only")
    p.add_argument("--hw", choices=list(HW_HBM_GB),
                   help="hardware (sets per-GPU HBM); default from launcher stem")
    p.add_argument("--quant", choices=list(QUANT_BYTES),
                   help="weight quantization (default from launcher/alias, else fp8)")
    p.add_argument("--mean-isl", type=int, default=None,
                   help="mean input tokens/request from your trace (default: from "
                        "the launcher's trace — 250000 if it sets the 256k WEKA "
                        "override, else 85000; 250000 for bare HF ids)")
    p.add_argument("--gpu-util", type=float, default=None,
                   help="HBM fraction the server reserves (default from launcher, else 0.9)")
    p.add_argument("--tp", type=int, default=None,
                   help="tensor-parallel size (default: the layouts configured in "
                        ".github/configs/amd-master.yaml, else 4)")
    p.add_argument("--dp", type=int, default=4, help="DP-attention ranks for DEP layout (default 4)")
    p.add_argument("--ep", type=int, default=8, help="expert-parallel size (default 8)")
    p.add_argument("--layout", choices=["tp", "tp-ep", "dep", "all"], default=None,
                   help="force a layout; default uses the configured layouts, "
                        "else 'all'")
    p.add_argument("--no-config", action="store_true",
                   help="ignore amd-master.yaml; use generic --tp/--layout sweep")
    # explicit overrides
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--attn-heads", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--kv-lora-rank", type=int)
    p.add_argument("--qk-rope-head-dim", type=int)
    args = p.parse_args()

    if args.list:
        with open(args.config_json) as fh:
            for m in sorted(json.load(fh)):
                print(m)
        return

    if not args.model:
        p.error("model is required (or use --list)")

    # Resolve alias / launcher path -> canonical model id (+ quant/hw hints).
    launcher_path = args.model if (args.model.endswith(".sh")
                                   and os.path.exists(args.model)) else None
    model_id, a_quant, a_hw = resolve_alias(args.model, here)
    was_alias = model_id != args.model
    args.model = model_id
    if a_hw and not args.hw:
        args.hw = a_hw
    if a_quant and not args.quant:
        args.quant = a_quant

    if args.quant is None:
        args.quant = "fp8"

    if not args.hw:
        sys.exit("error: --hw is required (or use a launcher/alias that implies "
                 "it), e.g. --hw mi355x")

    cfg = load_config(args.model, args)
    f = kv_formula(args.model, cfg)
    if f == "mla":
        required = ("num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim")
    elif f == "head_dim":
        required = ("num_hidden_layers", "num_key_value_heads", "head_dim")
    else:
        required = ("hidden_size", "num_attention_heads",
                    "num_hidden_layers", "num_key_value_heads")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        sys.exit(f"error: config for {args.model} ({f} formula) missing fields: {missing}")

    # Unless the user forced --tp/--layout/--no-config, default to the
    # parallelism layouts actually configured for this model+hw in
    # .github/configs/amd-master.yaml.
    specs, cfg_names = ([], [])
    forced = args.no_config or args.tp is not None or args.layout is not None
    if not forced:
        specs, cfg_names = configured_layouts(args.model, args.hw, here,
                                              quant=args.quant)

    # Inherit the run's real server knobs (util + trace ISL) from the launcher.
    # If invoked by a .sh path, use that; if by alias, locate the matching
    # agentic launcher (preferring the framework configured in amd-master.yaml).
    launch = parse_launcher(launcher_path) if launcher_path else {}
    if not launcher_path:
        found = find_launcher(args.model, args.quant, args.hw, here,
                              prefer=cfg_names)
        if found:
            launch = parse_launcher(found)
    if launch.get("gpu_util") and args.gpu_util is None:
        args.gpu_util = launch["gpu_util"]
    if args.mean_isl is None:
        args.mean_isl = launch.get("mean_isl", 250000)
    if args.dtype is None:
        if launch.get("kv_dtype") in DTYPE_BYTES:
            args.dtype = launch["kv_dtype"]
        else:
            # Agentic serving runs fp8 KV cache; default KV dtype to fp8 unless
            # the weights are a 16-bit format.
            args.dtype = "bfloat16" if args.quant in ("bf16", "bfloat16", "fp16",
                                                       "float16") else "fp8"
    if args.gpu_util is None:
        args.gpu_util = 0.9
    tp = args.tp if args.tp is not None else 4
    layout = args.layout or "all"
    layouts = ["tp", "tp-ep", "dep"] if layout == "all" else [layout]
    sim = simulate(args.model, cfg, args.dtype, args.quant, args.hw,
                   args.gpu_util, args.mean_isl,
                   layouts=layouts, tp=tp, dp=args.dp, ep=args.ep,
                   specs=specs or None)
    sim["source"] = ("amd-master.yaml: " + ", ".join(cfg_names)) if specs \
        else "generic sweep (no config match)"
    if args.json:
        print(json.dumps(sim, indent=2))
        return
    print(f"{sim['hw']} | {sim['model']} | KV={sim['dtype_kv']} "
          f"weights={sim['quant_weights']}")
    print(f"params~{sim['params_total_b']}B  weight~{sim['weight_gb_node']}GB(node)  "
          f"HBM {sim['hbm_gb']}GB x util {sim['gpu_util']}  "
          f"per-token KV {sim['per_token_kv_bytes']:,} B  "
          f"mean-ISL {sim['mean_isl']:,}")
    print(f"layouts from {sim['source']}")
    headers = ["layout", "TP", "DP", "EP", "weight/GPU", "KVcache/GPU",
               "crossover", "configured conc"]
    table_rows = []
    best = 0
    any_feasible = False
    for r in sim["rows"]:
        if "error" in r:
            table_rows.append([r["layout"], "", "", "", r["error"],
                               "", "", ""])
            continue
        cross = "OOM" if r.get("infeasible") else f"~{r['crossover_conc']:.0f}"
        conc = r.get("conc_list")
        if conc:
            conc_s = f"{conc[0]}…{conc[-1]}" if len(conc) > 1 else str(conc[0])
        else:
            conc_s = ""
        table_rows.append([
            r["layout"], str(r["tp"]), str(r["dp"]), str(r["ep"]),
            f"{r['weight_gb_per_gpu']} GB", f"{r['kv_pool_gb_per_gpu']} GB",
            cross, conc_s,
        ])
        if not r.get("infeasible"):
            any_feasible = True
            best = max(best, r["crossover_conc"])
    print(_box_table(headers, table_rows))
    print(
        "\ncolumns:\n"
        "  weight/GPU    weight memory per GPU = total_params x bytes/param / nGPU.\n"
        "                MoE = ALL experts resident. Fixed; KV fits in what's left.\n"
        "  KVcache/GPU   HBM left for KV per GPU = HBM x util - weight/GPU - ~8GB act.\n"
        "                This is the budget the KV cache lives in (what overflows).\n"
        "  crossover     concurrency where device KV overflows; above it,\n"
        "                hicache/lmcache offloading pays off.\n"
        "  configured    the conc-list swept for this layout in amd-master.yaml\n"
        "  conc          (first…last).")
    if not any_feasible:
        print(f"\nverdict: model does not fit at the given parallelism — "
              f"weights alone exceed the HBM budget (try a larger --tp). "
              f"'OOM' rows have no KV pool left.")
    else:
        print(f"\nverdict: hicache/lmcache offloading starts to pay off above "
              f"conc ~{best:.0f} (best feasible layout); below that the device "
              f"KV pool holds the working set.")
    print("note: crossover is the device-overflow knee for the given "
          "mean-ISL. Prefix-reuse hit-rate is a separate offloading benefit "
          "not modeled here.")


if __name__ == "__main__":
    main()
