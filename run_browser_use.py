"""
Run WebArena tasks using browser-use as the agent.

Drop-in replacement for run.py that swaps the WebArena prompt-agent loop for
a browser-use Agent while keeping the same task configs, auth refresh,
evaluator, and result directory layout.

Usage (called by run_eval_sglang.sh --agent-mode browser-use):
    python run_browser_use.py \
        --test_start_idx 0 --test_end_idx 12 \
        --model Qwen3-VL-8B-Instruct \
        --max_steps 30 \
        --result_dir results/browseruse_20260520_120000
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import random
from pathlib import Path

# ── WebArena imports ─────────────────────────────────────────────────────────
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.actions import create_stop_action
from evaluation_harness import evaluator_router
from evaluation_harness.helper_functions import PseudoPage
from playwright.sync_api import CDPSession


class PseudoCDPSession(CDPSession):
    """Stub CDPSession that satisfies beartype checks for evaluators that don't use CDP."""
    def send(self, method: str, params: dict = None) -> dict:
        return {}
    def detach(self) -> None:
        pass

# ── browser-use imports ──────────────────────────────────────────────────────
from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.openai.serializer import OpenAIMessageSerializer

# ── LLM logging ──────────────────────────────────────────────────────────────

_llm_log_file = None

def _get_llm_log_file():
    global _llm_log_file
    if _llm_log_file is None:
        result_dir = os.environ.get("WEBARENA_RESULT_DIR", ".")
        _llm_log_file = open(os.path.join(result_dir, "llm_logs.jsonl"), "a")
    return _llm_log_file


class LoggingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that logs every request/response pair to llm_logs.jsonl."""

    async def ainvoke(self, messages, output_format=None, **kwargs):
        result = await super().ainvoke(messages, output_format, **kwargs)
        try:
            serialized = OpenAIMessageSerializer.serialize_messages(messages)
            response = result.completion if hasattr(result, "completion") else str(result)
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            record = {
                "messages": serialized,
                "response": response,
            }
            _get_llm_log_file().write(json.dumps(record) + "\n")
            _get_llm_log_file().flush()
        except Exception as e:
            logger.warning(f"Failed to write LLM log: {e}")
        return result


# ── Logging setup ────────────────────────────────────────────────────────────
LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = (
    f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    f"_{random.randint(0, 10000)}.log"
)

logger = logging.getLogger("browser_use_runner")
logger.setLevel(logging.INFO)
for handler in [logging.StreamHandler(), logging.FileHandler(LOG_FILE_NAME)]:
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


# ── CLI ───────────────────────────────────────────────────────────────────────

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WebArena with browser-use agent")
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx",   type=int, default=12)
    parser.add_argument("--task_ids",       type=str, default="",
                        help="Comma-separated task IDs to run (overrides start/end idx)")
    parser.add_argument("--model",          type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--max_tokens",     type=int, default=2048)
    parser.add_argument("--max_steps",      type=int, default=30)
    parser.add_argument("--llm_timeout",    type=int, default=180,
                        help="Timeout in seconds for each LLM call (default 180)")
    parser.add_argument("--result_dir",     type=str, default="")
    return parser.parse_args()


# ── Auth helper ───────────────────────────────────────────────────────────────

def refresh_auth(task_config: dict) -> str | None:
    """Refresh site auth cookies and return path to the updated storage state."""
    if not task_config.get("storage_state"):
        return None
    cookie_file = os.path.basename(task_config["storage_state"])
    comb = get_site_comb_from_filepath(cookie_file)
    temp_dir = tempfile.mkdtemp()
    subprocess.run(
        [sys.executable, "browser_env/auto_login.py",
         "--auth_folder", temp_dir, "--site_list", *comb],
        check=False,
    )
    path = f"{temp_dir}/{cookie_file}"
    return path if os.path.exists(path) else None


# ── Single-task runner ────────────────────────────────────────────────────────

async def run_task(config_file: str, args: argparse.Namespace, llm: ChatOpenAI) -> float:
    with open(config_file) as f:
        task_config = json.load(f)

    intent    = task_config["intent"]
    task_id   = task_config["task_id"]
    start_url = task_config["start_url"]

    # Per-task log file — attach to root + all browser-use named loggers
    task_log_path = Path(args.result_dir) / f"task_{task_id}.log"
    task_handler = logging.FileHandler(task_log_path)
    task_handler.setLevel(logging.DEBUG)
    task_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    browser_use_loggers = ["browser_use", "Agent", "BrowserSession", "service", "browser_use_runner"]
    for log_name in browser_use_loggers:
        logging.getLogger(log_name).addHandler(task_handler)
    logging.getLogger().addHandler(task_handler)

    logger.info(f"[Config file]: {config_file}")
    logger.info(f"[Intent]: {intent}")

    # Refresh cookies
    storage_state = refresh_auth(task_config)

    # Update config file with fresh storage state so evaluator can find it
    if storage_state:
        task_config["storage_state"] = storage_state
        updated_config = f"{tempfile.mkdtemp()}/{os.path.basename(config_file)}"
        with open(updated_config, "w") as f:
            json.dump(task_config, f)
        config_file = updated_config

    # Build browser-use agent
    # Restrict navigation to WebArena hosts only — prevents the model from
    # doing external searches (Google, Bing, etc.) or hallucinating external URLs.
    browser_profile = BrowserProfile(
        headless=True,
        storage_state=storage_state,
        allowed_domains=["158.130.4.228", "158.130.4.229"],
    )

    # Navigate to start_url programmatically before the LLM gets involved,
    # so the model never sees or has to reproduce the URL.
    task_prompt = (
        f"{intent}\n"
        f"When you have the final answer, use the done action and state it clearly."
    )

    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser_profile=browser_profile,
        initial_actions=[{"navigate": {"url": start_url}}],
        llm_timeout=args.llm_timeout,
        use_judge=False,
    )

    final_answer = ""
    score = 0.0

    try:
        history = await agent.run(max_steps=args.max_steps)
        final_answer = history.final_result() or ""

        # Evaluate while browser is still open so evaluators can use the live page
        try:
            live_page = await agent.browser_session.get_current_page()
            final_url = live_page.url if live_page else start_url
            pseudo_page = PseudoPage(live_page, final_url)
            pseudo_client = PseudoCDPSession.__new__(PseudoCDPSession)
            trajectory = [create_stop_action(final_answer)]
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=pseudo_page,
                client=pseudo_client,
            )
        except Exception as e:
            logger.warning(f"[evaluator error] task {task_id}: {e}")
            score = 0.0

    except Exception as e:
        logger.warning(f"[browser-use error] task {task_id}: {e}")
        final_answer = ""
        score = 0.0

    # Close browser session
    try:
        await agent.browser_session.stop()
    except Exception:
        pass

    # Remove per-task log handler from all loggers
    for log_name in browser_use_loggers:
        logging.getLogger(log_name).removeHandler(task_handler)
    logging.getLogger().removeHandler(task_handler)
    task_handler.close()

    # Log result
    if score == 1:
        logger.info(f"[Result] (PASS) {config_file}")
    else:
        logger.info(f"[Result] (FAIL) {config_file}  answer={final_answer!r}")

    # Save result JSON
    result_path = Path(args.result_dir) / f"result_{task_id}.json"
    result_path.write_text(json.dumps({
        "task_id": task_id,
        "intent": intent,
        "final_answer": final_answer,
        "score": score,
    }, indent=2))

    return score


# ── Main loop ─────────────────────────────────────────────────────────────────

def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    done_ids = {
        Path(f).stem.split("_")[1]
        for f in glob.glob(f"{result_dir}/result_*.json")
    }
    return [f for f in config_files
            if Path(f).stem not in done_ids]


async def main() -> None:
    args = config()

    # Result dir
    if not args.result_dir:
        args.result_dir = f"results/browseruse_{time.strftime('%Y%m%d%H%M%S')}"
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # Log which log file we're using
    with open(Path(args.result_dir) / "log_files.txt", "a") as f:
        f.write(f"{LOG_FILE_NAME}\n")

    # Save config
    config_out = Path(args.result_dir) / "config.json"
    if not config_out.exists():
        config_out.write_text(json.dumps(vars(args), indent=4))

    # Build task list
    if args.task_ids:
        ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
        config_files = [f"config_files/{i}.json" for i in ids]
    else:
        config_files = [f"config_files/{i}.json"
                        for i in range(args.test_start_idx, args.test_end_idx)]
    config_files = get_unfinished(config_files, args.result_dir)

    if not config_files:
        logger.info("No tasks left to run.")
        return

    logger.info(f"Total tasks to run: {len(config_files)}")

    # LLM — connects to SGLang via OPENAI_API_BASE env var (set in shell script)
    llm = LoggingChatOpenAI(
        model=args.model,
        base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=args.temperature,
        max_completion_tokens=args.max_tokens,
    )

    scores = []
    for config_file in config_files:
        score = await run_task(config_file, args, llm)
        scores.append(score)

    if scores:
        logger.info(
            f"Done. {len(scores)} tasks | "
            f"passed: {sum(scores)} | "
            f"success rate: {sum(scores)/len(scores)*100:.1f}%"
        )


if __name__ == "__main__":
    asyncio.run(main())
