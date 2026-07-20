"""Orchestrate a batch of WebArena tasks at high parallelism.

Site-aware scheduler: keeps every available (site, replica) slot busy and NEVER idle-blocks
a worker on a busy single-replica site (e.g. map). A task is dispatched only when a free
replica exists for each of its sites; single-replica sites (map/wiki/homepage) naturally
serialize but only among themselves — they never stall other sites.

LLM load balancing: spreads requests round-robin across N single-GPU model replicas
(--base-urls url1,url2,...), so all 4 H100s are used (one independent Qwen3-32B each).

Each task runs as its own run_task.py subprocess. Conn-fail tasks are re-queued (retry) on
a fresh proxy. Writes the resolved task (eval + replica_map) to <out>/task_<id>/input.json.
"""
import os, sys, json, time, argparse, subprocess, threading
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wa_config as C

CONN_MARKERS = ("ERR_TIMED_OUT", "ERR_PROXY", "ERR_CONNECTION", "ERR_SOCKS", "ERR_NETWORK",
                "No internet", "ERR_NAME_NOT", "proxy connection")

# Read-only sites: agents never mutate server state, so many tasks may safely share the
# single replica concurrently (no isolation needed). Mutating sites stay at 1 task/replica.
READONLY_SITES = {"map", "wikipedia", "homepage"}
RO_CAP = 16  # concurrent tasks per read-only replica (map/wiki/homepage)


def is_conn_fail(out_task_dir, tid):
    try:
        r = json.load(open(os.path.join(out_task_dir, f"task_{tid}.json")))
    except Exception:
        return True
    ans = (r.get("answer") or ""); err = (r.get("error") or "")
    real_urls = sum(1 for s in r.get("steps", []) if s.get("url") and "about:" not in s["url"])
    if any(m.lower() in (ans + err).lower() for m in CONN_MARKERS):
        return real_urls < 3
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--base-urls", default="", help="comma-sep LLM replicas for load balancing")
    ap.add_argument("--model", default="qwen3vl-dense")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--num-proxies", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--task-timeout", type=float, default=1500.0)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--workflows", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_urls = [u for u in args.base_urls.split(",") if u.strip()] or [args.base_url]

    raw = C.load_raw_tasks()
    by_id = {t["task_id"]: t for t in raw}
    if args.only:
        ids = [int(x) for x in args.only.split(",") if x.strip() != ""]
    else:
        ids = [i for i in range(args.start, args.end) if i in by_id]
    if args.skip_done:
        ids = [i for i in ids if not os.path.exists(os.path.join(args.out_dir, f"task_{i}", f"task_{i}.json"))]

    # scheduler state: per-(site,replica) remaining slot capacity
    # read-only sites get RO_CAP concurrent per replica; mutating sites get 1.
    slots = {}
    for s in C.PORTS:
        cap = RO_CAP if s in READONLY_SITES else 1
        for ri in range(len(C.PORTS[s])):
            slots[(s, ri)] = cap
    pending = deque(by_id[i] for i in ids)
    attempts = {i: 0 for i in ids}
    cond = threading.Condition()
    inflight = [0]
    done = [0]
    proxy_ctr = [0]

    print(f"running {len(ids)} tasks, max_concurrency={args.max_concurrency}, "
          f"LLM replicas={len(base_urls)}, proxies={args.num_proxies} -> {args.out_dir}", flush=True)
    t0 = time.time()

    def worker(task, assign, base_url, proxy_port):
        tid = task["task_id"]
        sites = task["sites"]
        out_task_dir = os.path.join(args.out_dir, f"task_{tid}")
        os.makedirs(out_task_dir, exist_ok=True)
        resolved = C.resolve_task(task, assign); resolved["replica_map"] = assign
        json.dump(resolved, open(os.path.join(out_task_dir, "input.json"), "w"),
                  ensure_ascii=False, indent=2)
        ts = time.time()
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_task.py"),
               "--task-json", os.path.join(out_task_dir, "input.json"), "--out-dir", args.out_dir,
               "--base-url", base_url, "--model", args.model,
               "--proxy", f"http://127.0.0.1:{proxy_port}",
               "--max-steps", str(args.max_steps), "--max-tokens", str(args.max_tokens),
               "--temperature", str(args.temperature), "--task-timeout", str(args.task_timeout)]
        if args.no_think:
            cmd.append("--no-think")
        if args.workflows:
            cmd += ["--workflows", args.workflows]
        try:
            with open(os.path.join(out_task_dir, "run.log"), "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.task_timeout + 180)
        except subprocess.TimeoutExpired:
            pass
        cf = is_conn_fail(out_task_dir, tid) and attempts[tid] < args.retries
        with cond:
            for s, ri in assign.items():
                slots[(s, ri)] += 1
            inflight[0] -= 1
            if cf:
                attempts[tid] += 1
                pending.append(task)
                print(f"[{tid}] conn-fail, requeue (attempt {attempts[tid]})", flush=True)
            else:
                done[0] += 1
                try:
                    r = json.load(open(os.path.join(out_task_dir, f"task_{tid}.json")))
                    st = f"done={r.get('is_done')} steps={r.get('n_steps')} err={r.get('error')}"
                except Exception:
                    st = "NO_RESULT"
                print(f"[{tid}] {sites} rep={assign} {round(time.time()-ts,1)}s {st} "
                      f"[{done[0]}/{len(ids)} elapsed={round(time.time()-t0)}s]", flush=True)
            cond.notify_all()

    with cond:
        while pending or inflight[0] > 0:
            progressed = True
            while progressed:
                progressed = False
                for task in list(pending):
                    if inflight[0] >= args.max_concurrency:
                        break
                    sites = task["sites"]
                    # find a replica with a free slot for each site (pick most-free = load balance)
                    assign = {}
                    ok = True
                    for s in sites:
                        cand = max(range(len(C.PORTS[s])), key=lambda ri: slots[(s, ri)])
                        if slots[(s, cand)] > 0:
                            assign[s] = cand
                        else:
                            ok = False
                            break
                    if ok:
                        for s, ri in assign.items():
                            slots[(s, ri)] -= 1
                        pending.remove(task)
                        inflight[0] += 1
                        bu = base_urls[proxy_ctr[0] % len(base_urls)]
                        pp = 18900 + (proxy_ctr[0] % args.num_proxies)
                        proxy_ctr[0] += 1
                        threading.Thread(target=worker, args=(task, assign, bu, pp), daemon=True).start()
                        progressed = True
            cond.wait()

    print(f"BATCH DONE {len(ids)} tasks in {round(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
