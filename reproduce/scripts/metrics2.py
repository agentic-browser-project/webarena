"""Sparsity metrics for the exp2 (sr_compare-format) results.

Reads each task's llm_calls.jsonl, reconstructs the prompt actually sent to the
server (last call = largest context), and computes:
  - input_len (tokens)
  - TSA: num_chunks via the SAME TreeSparseSelector/webarena parse the server uses,
         selected_chunks = min(top_k, num_chunks) -> %chunks
         num_pages(page_size=64) for reference
  - quest: num 16-token pages, selected = min(61, npages) -> %pages (exact)
Usage: metrics2.py <method_dir> <mode:tsa|quest|dense> [top_k]
"""
import sys, os, json, glob
import numpy as np
_TSA = os.environ.get("TSA_REPO", "/home/cc/TreeSparseAttention")
sys.path[:0] = [_TSA, _TSA + "/models", _TSA + "/python"]
from transformers import AutoTokenizer

D, MODE = sys.argv[1], sys.argv[2]
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
MODEL = os.environ.get("MODEL_PATH", "/home/cc/hf_models/Qwen3-VL-32B-Instruct")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def msgs_to_text(msgs):
    parts = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
    return "\n".join(parts)


sel = None
if MODE == "tsa":
    from tree_sparse_selector import TreeSparseSelector, TreeSparseConfig
    sel = TreeSparseSelector(tok,
                             config=TreeSparseConfig(top_k_chunks=TOPK, tree_parse_mode="webarena"),
                             num_layers=64, num_qo_heads=64, num_kv_heads=8,
                             head_dim=128, device="cpu")

ilens, nch, pctch, npg, pctpg, outs = [], [], [], [], [], []
for f in sorted(glob.glob(f"{D}/task_*/llm_calls.jsonl")):
    last = None
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        last = r
        u = r.get("usage") or {}
        if u.get("completion_tokens"):
            outs.append(u["completion_tokens"])
    if not last:
        continue
    text = msgs_to_text(last.get("input_messages") or [])
    ids = tok(text).input_ids
    ilen = len(ids)
    if ilen < 50:
        continue
    ilens.append(ilen)
    npg.append((ilen + 63) // 64)
    if MODE == "tsa":
        try:
            sel.register_request(ids)
            n = len(sel.chunks)
        except Exception:
            n = 0
        if n:
            nch.append(n)
            pctch.append(100.0 * min(TOPK, n) / n)
    elif MODE == "quest":
        nq = (ilen + 15) // 16
        npg[-1] = nq
        pctpg.append(100.0 * min(61, nq) / nq)

out = {
    "dir": D, "mode": MODE, "top_k": TOPK, "n_tasks": len(ilens),
    "median_input_len": int(np.median(ilens)) if ilens else 0,
    "mean_input_len": round(float(np.mean(ilens)), 1) if ilens else 0,
    "median_output_len": int(np.median(outs)) if outs else 0,
}
if MODE == "tsa":
    out["median_num_chunks"] = int(np.median(nch)) if nch else 0
    out["mean_pct_chunks"] = round(float(np.mean(pctch)), 1) if pctch else None
elif MODE == "quest":
    out["median_num_pages_16"] = int(np.median(npg)) if npg else 0
    out["mean_pct_pages"] = round(float(np.mean(pctpg)), 1) if pctpg else None
print(json.dumps(out))
