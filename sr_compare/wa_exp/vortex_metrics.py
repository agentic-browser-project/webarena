"""Post-hoc vortex page-selection metrics (selected/total KV blocks per LLM inference).
Budget per inference is deterministic from the prompt length + vortex config:
    total_blocks    = ceil(prompt_tokens / block_size)            (block_size = page_size = 16)
    dynamic         = ceil(topk_ratio * total_blocks)             (ratio mode)
    selected_blocks = min(total_blocks, max(topk_val, dynamic) + reserved_bos+eos(=2))
This gives the exact selected/total page ratio. (Which blocks vary per layer; the COUNT does
not, so the ratio is exact.) Reads results/<dir>/task_*/llm_calls.jsonl -> sparse_metrics_vortex.jsonl
Usage: python vortex_metrics.py <results_dir> [topk_val=30] [block_size=16] [topk_ratio=0.0] [reserved=2]
"""
import sys, json, glob, os, math

def main():
    rdir = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    block = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    reserved = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    out = os.path.join(rdir, "sparse_metrics_vortex.jsonl")
    n = 0
    with open(out, "w") as w:
        for f in sorted(glob.glob(os.path.join(rdir, "task_*/llm_calls.jsonl"))):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                u = r.get("usage") or {}
                pt = u.get("prompt_tokens") or u.get("input_tokens")
                if not pt:
                    continue
                total = max(1, math.ceil(pt / block))
                dynamic = math.ceil(ratio * total)
                sel = min(total, max(topk, dynamic) + reserved)
                w.write(json.dumps({
                    "tag": "vortex", "prompt_tokens": pt,
                    "selected_pages_mean": sel, "total_pages": total,
                    "page_ratio": round(sel / total, 4),
                    "vortex_topk": topk, "topk_ratio": ratio, "block_size": block,
                }) + "\n")
                n += 1
    print(f"[vortex_metrics] wrote {n} inferences -> {out}")

if __name__ == "__main__":
    main()
