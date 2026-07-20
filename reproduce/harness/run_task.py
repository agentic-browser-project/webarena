"""Run ONE WebArena task with browser-use against an OpenAI-compatible LLM endpoint.

Captures: full LLM input trajectory + output + token usage (llm_calls.jsonl), per-step
browser trajectory, final answer, final_url -> task_<id>.json. Backend-agnostic: point
--base-url at sglang dense, TSA, or vortex.
"""
import os, sys, json, time, argparse, asyncio, traceback, threading
# Heavy WebArena sites (Magento admin/storefront) load slowly through the SSH SOCKS proxy;
# raise browser-use's per-event timeouts (read from env at import time).
os.environ.setdefault("TIMEOUT_NavigateToUrlEvent", "90")
os.environ.setdefault("TIMEOUT_ClickElementEvent", "45")
os.environ.setdefault("TIMEOUT_RefreshEvent", "45")
os.environ.setdefault("TIMEOUT_GoBackEvent", "45")
os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", "60")
os.environ.setdefault("TIMEOUT_ScrollToTextEvent", "30")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wa_config as C
from browser_use import Agent, ChatOpenAI, BrowserSession, BrowserProfile

# ---------- per-process LLM call logger ----------
_LLM_LOCK = threading.Lock()
LLM_LOG = None
_CALL = [0]

def _ser_msg(m):
    try:
        return m.model_dump(mode="json")
    except Exception:
        return {"role": getattr(m, "role", None), "content": str(getattr(m, "content", m))}

def _as_list(c):
    """Normalize message content to a list of content-parts (preserves images)."""
    if isinstance(c, list):
        return c
    return [{"type": "text", "text": c if isinstance(c, str) else str(c)}]

def merge_consecutive_same_role(messages):
    """Merge consecutive same-role messages into one. Needed for models whose chat template
    enforces strict user/assistant alternation (e.g. Gemma raises 'roles must alternate' 400)
    when browser-use injects extra UserMessage nudges. Model-agnostic; harmless for tolerant
    models (Llama/Qwen). Preserves list/image content parts."""
    merged = []
    for m in messages:
        if merged and getattr(merged[-1], "role", None) == getattr(m, "role", None):
            prev = merged[-1]
            a, b = prev.content, m.content
            if isinstance(a, str) and isinstance(b, str):
                newc = a + "\n\n" + b
            else:
                newc = _as_list(a) + _as_list(b)
            try:
                merged[-1] = prev.model_copy(update={"content": newc})
            except Exception:
                prev.content = newc
        else:
            merged.append(m)
    return merged

class LoggingChatOpenAI(ChatOpenAI):
    async def ainvoke(self, messages, output_format=None, **kwargs):
        t0 = time.time()
        messages = merge_consecutive_same_role(messages)
        res = await super().ainvoke(messages, output_format, **kwargs)
        try:
            out = res.completion
            out_s = out.model_dump(mode="json") if hasattr(out, "model_dump") else str(out)
            usage = res.usage.model_dump() if getattr(res, "usage", None) is not None else None
            rec = {"call": _CALL[0], "t": t0, "latency_s": round(time.time() - t0, 3),
                   "input_messages": [_ser_msg(m) for m in messages],
                   "output": out_s, "usage": usage}
            _CALL[0] += 1
            if LLM_LOG:
                with _LLM_LOCK, open(LLM_LOG, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            traceback.print_exc()
        return res


import subprocess, socket

HTTP_PROXY_PORTS = list(range(18900, 18908))

def pick_http_proxy(preferred, n_ports=8):
    """Pick a currently-working HTTP proxy (hilbit2 remote proxy via ssh -L), preferring
    the assigned one. These are robust: chromium pools keep-alive connections to them."""
    # This machine has no HTTP proxies (18900-8); it reaches the sites through a single
    # SSH SOCKS proxy. Force it when WA_FORCE_PROXY is set (chromium/playwright supports socks5).
    forced = os.environ.get("WA_FORCE_PROXY")
    if forced:
        return forced
    try:
        pref = int(preferred.rsplit(":", 1)[1])
    except Exception:
        pref = 18900
    order = [pref] + [p for p in HTTP_PROXY_PORTS if p != pref]
    for port in order:
        try:
            code = subprocess.run(
                ["curl", "-s", "-m", "12", "-x", f"http://127.0.0.1:{port}",
                 "-o", "/dev/null", "-w", "%{http_code}", "http://158.130.4.158:7780/admin"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            if code in ("200", "302"):
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return preferred


def merge_storage_state(login_sites, replica_map):
    """Merge cookies+origins from each login site's auth file into one storage_state dict."""
    cookies, origins = [], []
    for site in login_sites:
        ri = replica_map.get(site, 0)
        p = C.auth_path(site, ri)
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        cookies.extend(st.get("cookies", []))
        origins.extend(st.get("origins", []))
    return {"cookies": cookies, "origins": origins}


async def run(task, args):
    global LLM_LOG
    tid = task["task_id"]
    out_dir = os.path.join(args.out_dir, f"task_{tid}")
    os.makedirs(out_dir, exist_ok=True)
    LLM_LOG = os.path.join(out_dir, "llm_calls.jsonl")
    open(LLM_LOG, "w").close()

    replica_map = task.get("replica_map", {})
    login_sites = [s for s in task["sites"] if s in C.ACCOUNTS]
    storage_state = None
    if login_sites:
        ss = merge_storage_state(login_sites, replica_map)
        storage_state = os.path.join(out_dir, "storage_state.json")
        json.dump(ss, open(storage_state, "w"))

    llm = LoggingChatOpenAI(
        model=args.model, base_url=args.base_url, api_key="EMPTY",
        temperature=args.temperature, max_completion_tokens=args.max_tokens, timeout=args.llm_timeout)

    proxy_url = pick_http_proxy(args.proxy)
    profile = BrowserProfile(
        headless=True, proxy={"server": proxy_url},
        storage_state=storage_state, allowed_domains=set(C.all_hosts()),
        block_ip_addresses=False, chromium_sandbox=False,
        args=["--blink-settings=imagesEnabled=false", "--no-sandbox", "--disable-dev-shm-usage"],
        keep_alive=False, minimum_wait_page_load_time=0.8,
        wait_for_network_idle_page_load_time=1.5, maximum_wait_page_load_time=30.0,
        user_data_dir=None)
    session = BrowserSession(browser_profile=profile)

    steps_log = []
    async def step_cb(browser_state, agent_output, step_number):
        try:
            ng = getattr(getattr(agent_output, "current_state", None), "next_goal", None)
            th = getattr(getattr(agent_output, "current_state", None), "thinking", None)
            acts = [a.model_dump(exclude_none=True, mode="json") for a in (agent_output.action or [])]
            steps_log.append({"step": step_number, "t": time.time(),
                              "url": getattr(browser_state, "url", None),
                              "next_goal": ng, "thinking": th, "actions": acts})
        except Exception:
            traceback.print_exc()

    # ---- AWM: inject same-site workflows learned from OTHER tasks' successes (leave-one-out) ----
    wf_section = ""
    if getattr(args, "workflows", ""):
        try:
            allwf = json.load(open(args.workflows))
            mysites = set(task.get("sites", []))
            rel = [w for w in allwf if w.get("task_id") != tid and (set(w.get("sites", [])) & mysites)]
            rel = rel[:4]  # cap to keep prompt size reasonable
            if rel:
                wf_section = ("LEARNED WORKFLOWS (reusable navigation procedures distilled from past "
                              "SUCCESSFUL tasks on this site — adapt them, they may not match exactly):\n")
                for i, w in enumerate(rel, 1):
                    wf_section += f"[Workflow {i}] {w['workflow']}\n\n"
        except Exception:
            traceback.print_exc()

    preamble = (
        "IMPORTANT ENVIRONMENT NOTES:\n"
        "- You operate inside a self-hosted sandbox that mirrors real websites. The current "
        f"browser tab is ALREADY on the correct target site for this task: {task['start_url']}\n"
        "- Do NOT navigate to the public internet (real reddit.com, github.com, google.com, "
        "bing.com, openstreetmap.org, amazon.com, etc.). They are NOT part of this task and are "
        "blocked. A blocked/empty page does NOT mean the site is down — it means you left the "
        "sandbox. Stay on the current site.\n"
        "- Do NOT use the search_google/web-search action. Find information using THIS site's own "
        "navigation menus, search boxes, and links, starting from the current page.\n"
        "- The sandbox sites are: Shopping (OneStopMarket), Shopping Admin (Magento backend), a "
        "Reddit-like forum (Postmill), GitLab, a Map, and Wikipedia.\n"
        "- If the task is genuinely impossible on this site, say so explicitly in your final answer.\n\n"
        + wf_section +
        f"YOUR TASK: {task['intent']}"
    )
    if args.no_think:
        preamble = "/no_think\n" + preamble
    result = {"task_id": tid, "intent": task["intent"], "start_url": task["start_url"],
              "sites": task["sites"], "replica_map": replica_map, "model": args.model,
              "base_url": args.base_url, "t_start": time.time()}
    try:
        agent = Agent(
            task=preamble, llm=llm, browser_session=session, use_vision=False,
            llm_timeout=args.llm_call_timeout, step_timeout=args.step_timeout,
            use_judge=False, max_actions_per_step=args.max_actions,
            initial_actions=[{"navigate": {"url": task["start_url"], "new_tab": False}}],
            register_new_step_callback=step_cb, calculate_cost=False)
        hist = await asyncio.wait_for(agent.run(max_steps=args.max_steps), timeout=args.task_timeout)
        result["answer"] = hist.final_result()
        urls = [u for u in hist.urls() if u]
        result["final_url"] = urls[-1] if urls else None
        result["n_steps"] = hist.number_of_steps()
        result["is_done"] = hist.is_done()
        result["self_report_success"] = hist.is_successful()
        result["errors"] = hist.errors()
        result["error"] = None
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        result["answer"] = None
        result["final_url"] = None
        result["n_steps"] = len(steps_log)
        traceback.print_exc()
    finally:
        try:
            await session.kill()
        except Exception:
            pass
    result["t_end"] = time.time()
    result["wall_time_s"] = round(result["t_end"] - result["t_start"], 2)
    result["steps"] = steps_log
    with open(os.path.join(out_dir, f"task_{tid}.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[task {tid}] done={result.get('is_done')} steps={result.get('n_steps')} "
          f"err={result.get('error')} answer_len={len(result.get('answer') or '')}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default="qwen3vl-dense")
    ap.add_argument("--proxy", default="http://127.0.0.1:18900")
    ap.add_argument("--no-think", action="store_true", help="prepend /no_think (Qwen3 hybrid reasoning off)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-actions", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout", type=float, default=400.0)
    ap.add_argument("--task-timeout", type=float, default=1200.0)
    ap.add_argument("--workflows", type=str, default="")
    ap.add_argument("--llm-call-timeout", dest="llm_call_timeout", type=int, default=300)
    ap.add_argument("--step-timeout", dest="step_timeout", type=int, default=500)
    args = ap.parse_args()
    task = json.load(open(args.task_json))
    asyncio.run(run(task, args))


if __name__ == "__main__":
    main()
