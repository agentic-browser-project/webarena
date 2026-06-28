"""Score ONE WebArena task with the UNMODIFIED evaluator_router.

Reads the resolved task config (input.json, contains eval + URLs for the replica the task
ran on) and the agent result (task_<id>.json). Opens an authenticated Playwright page
(through the SOCKS proxy), navigates to the agent's final URL, and runs the evaluator.
The LLM fuzzy/ua judge is repointed to the local dense server (same judge for all methods).

Env URLs (SHOPPING/REDDIT/...) must be set by the caller to the task's replica URLs so that
program_html `func:` helpers resolve correctly.
"""
import os, sys, json, argparse, types
sys.path.insert(0, "/home/cc/temp/webarena/sr_compare/wa_exp")
import wa_config as C

# webarena's llms provider imports the legacy openai 0.x API; we never call it (judge is
# patched to urllib), so shim the missing module so the import chain succeeds on openai 2.x.
import openai
if not hasattr(openai, "error"):
    _err = types.ModuleType("openai.error")
    for _n in ["OpenAIError", "RateLimitError", "APIError", "APIConnectionError", "Timeout",
               "ServiceUnavailableError", "InvalidRequestError", "AuthenticationError"]:
        setattr(_err, _n, type(_n, (Exception,), {}))
    sys.modules["openai.error"] = _err
    openai.error = _err

WEBARENA = "/home/cc/webarena"
sys.path.insert(0, WEBARENA)
os.chdir(WEBARENA)

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:18000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3vl-dense")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")


def _judge_chat(messages, temperature=0, max_tokens=768):
    import urllib.request
    body = json.dumps({"model": JUDGE_MODEL, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(JUDGE_BASE_URL.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {JUDGE_API_KEY}"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return r["choices"][0]["message"]["content"]


import evaluation_harness.evaluators as _ev

def _llm_fuzzy_match(pred, reference, question):
    msg = ("Help a teacher to grade the answer of a student given a question. Keep in mind that "
           "the student may use different phrasing or wording to answer the question. The goal is "
           "to evaluate whether the answer is semantically equivalent to the reference answer.\n"
           f"question: {question}\nreference answer: {reference}\n"
           "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
           f"student answer: {pred}\nConclude the judgement by correct/incorrect/partially correct.")
    resp = _judge_chat([{"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": msg}]).lower()
    return 0.0 if ("partially correct" in resp or "incorrect" in resp) else 1.0

def _llm_ua_match(pred, reference, question):
    msg = ("Given a task that asks whether a goal is achievable and a student answer, judge whether "
           "the student correctly identified achievability.\n"
           f"task: {question}\nreference (reason it is not achievable): {reference}\n"
           f"student answer: {pred}\nConclude the judgement by correct/incorrect.")
    resp = _judge_chat([{"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": msg}]).lower()
    return 0.0 if "incorrect" in resp else 1.0

_ev.llm_fuzzy_match = _llm_fuzzy_match
_ev.llm_ua_match = _llm_ua_match

from browser_env.actions import create_stop_action
from evaluation_harness import evaluator_router
from playwright.sync_api import sync_playwright


def storage_for(task):
    login_sites = [s for s in task["sites"] if s in C.ACCOUNTS]
    rmap = task.get("replica_map", {})
    cookies = []
    for site in login_sites:
        p = C.auth_path(site, rmap.get(site, 0))
        if os.path.exists(p):
            cookies.extend(json.load(open(p)).get("cookies", []))
    return {"cookies": cookies, "origins": []} if cookies else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    args = ap.parse_args()
    tid = int(os.path.basename(args.task_dir.rstrip("/")).replace("task_", ""))
    task = json.load(open(os.path.join(args.task_dir, "input.json")))
    res = json.load(open(os.path.join(args.task_dir, f"task_{tid}.json")))
    answer = res.get("answer") or ""
    final_url = res.get("final_url")

    # write resolved config to a path evaluator_router can read
    cfg_path = os.path.join(args.task_dir, "input.json")
    ss = storage_for(task)

    pw = sync_playwright().start()
    _proxy = os.environ.get("SCORE_PROXY", "http://127.0.0.1:18900")  # C.PROXY (socks 1080) is dead
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"],
                                 proxy={"server": _proxy})
    out = {"task_id": tid, "score": None, "nav_ok": None,
           "eval_types": task["eval"]["eval_types"], "error": None}
    try:
        ctx = browser.new_context(storage_state=ss)
        page = ctx.new_page()
        client = ctx.new_cdp_session(page)
        nav_ok = True
        if final_url and not str(final_url).startswith("about:"):
            try:
                page.goto(final_url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                nav_ok = False
                print(f"[scorer {tid}] goto failed: {e}", file=sys.stderr)
        trajectory = [create_stop_action(answer)]
        evaluator = evaluator_router(cfg_path)
        score = evaluator(trajectory, cfg_path, page, client)
        out["score"] = float(score)
        out["nav_ok"] = nav_ok
        ctx.close()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        browser.close()
        pw.stop()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
