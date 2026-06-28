"""Aggregate per-inference sparse-selection metrics for one config.
Reads all sparse_metrics*.jsonl under the config results dir (TSA writes one per GPU
replica; vortex writes its own). Prints + saves a summary: mean selected/total chunks
and pages across all logged LLM inferences.
Usage: python agg_sparse_metrics.py /home/cc/temp/webarena/sr_compare/wa_exp/results/<config>
"""
import sys, json, glob, os, statistics as st

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "sparse_metrics*.jsonl")))
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    # drop warmup/dummy samples (tiny contexts) that skew the mean (e.g. cuda-graph capture bs=1)
    rows = [r for r in rows if (r.get("total_pages") or 0) >= 10 or (r.get("total_chunks") or 0) >= 4]
    if not rows:
        print(f"[agg] no sparse metrics found under {d}")
        return
    def col(k):
        return [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    cr, pr = col("chunk_ratio"), col("page_ratio")
    sc, tc = col("selected_chunks"), col("total_chunks")
    sp, tp = col("selected_pages_mean"), col("total_pages")
    summary = {
        "config": os.path.basename(d.rstrip("/")),
        "n_inferences": len(rows),
        "chunk_ratio_mean": round(st.mean(cr), 4) if cr else None,
        "chunk_ratio_median": round(st.median(cr), 4) if cr else None,
        "page_ratio_mean": round(st.mean(pr), 4) if pr else None,
        "page_ratio_median": round(st.median(pr), 4) if pr else None,
        "selected_chunks_mean": round(st.mean(sc), 2) if sc else None,
        "total_chunks_mean": round(st.mean(tc), 2) if tc else None,
        "selected_pages_mean": round(st.mean(sp), 2) if sp else None,
        "total_pages_mean": round(st.mean(tp), 2) if tp else None,
    }
    out = os.path.join(d, "sparse_metrics_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[agg] wrote {out}")

if __name__ == "__main__":
    main()
