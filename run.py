"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")

    # multi-worker support
    parser.add_argument(
        "--worker_id",
        type=int,
        default=None,
        help="If set, this run.py invocation is one worker of a multi-worker "
        "harness; URL env vars are loaded from mp/config.json for this id and "
        "the auth folder is per-worker.",
    )
    parser.add_argument(
        "--mp_config_path",
        type=str,
        default=None,
        help="Optional override for mp/config.json path (used with --worker_id).",
    )

    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def _apply_worker_env(args: argparse.Namespace) -> None:
    """If --worker_id is set, override URL env vars and the auth folder root
    to match this worker's per-replica URLs and per-worker auth state."""
    if args.worker_id is None:
        return
    from mp.config import load_config

    cfg = load_config(args.mp_config_path)
    for k, v in cfg.env_for(args.worker_id).items():
        os.environ[k] = v
    # Per-worker auth folder — used by run_single_task instead of the global .auth dir.
    os.environ["WEBARENA_AUTH_FOLDER"] = f"{cfg.auth_root}/w{args.worker_id}"
    os.environ["WEBARENA_WORKER_ID"] = str(args.worker_id)
    os.makedirs(os.environ["WEBARENA_AUTH_FOLDER"], exist_ok=True)


def run_single_task(
    config_file: str,
    *,
    worker_id: int,
    cfg,
    args_dict: dict[str, object],
) -> float:
    """Run one task end-to-end and return the score.

    Adapter for ``mp.worker._run_one_task`` so a worker process can call
    the same agent/evaluator machinery a serial ``run.py --test_start_idx``
    invocation uses. Returns 0.0 on agent failure and re-raises on
    catastrophic env errors.
    """
    # Build an argparse-namespace clone from args_dict so the function below
    # can be called with the same shape as `test()` would receive.
    ns = argparse.Namespace(**args_dict)
    ns.render = False
    ns.render_screenshot = True
    ns.save_trace_enabled = bool(args_dict.get("save_trace_enabled", True))
    ns.current_viewport_only = True
    ns.worker_id = worker_id
    ns.mp_config_path = None
    ns.sleep_after_execution = 2.0
    ns.parsing_failure_th = int(args_dict.get("parsing_failure_th", 3))  # type: ignore[arg-type,call-overload]
    ns.repeating_action_failure_th = int(args_dict.get("repeating_action_failure_th", 3))  # type: ignore[arg-type,call-overload]
    ns.slow_mo = int(args_dict.get("slow_mo", 0))  # type: ignore[arg-type,call-overload]
    ns.context_length = int(args_dict.get("context_length", 0))  # type: ignore[arg-type,call-overload]
    ns.max_retry = int(args_dict.get("max_retry", 1))  # type: ignore[arg-type,call-overload]
    ns.max_obs_length = int(args_dict.get("max_obs_length", 1920))  # type: ignore[arg-type,call-overload]
    ns.model_endpoint = args_dict.get("model_endpoint", "")
    ns.stop_token = args_dict.get("stop_token", None)
    # results dir under cfg.result_dir/w{worker_id}
    ns.result_dir = f"{cfg.result_dir}/w{worker_id}"
    os.makedirs(ns.result_dir, exist_ok=True)
    os.makedirs(f"{ns.result_dir}/traces", exist_ok=True)

    # The agent constructor reads stuff from disk; needs prompt JSON.
    from agent.prompts import to_json
    to_json.run()

    agent = construct_agent(ns)

    # Reuse the body of `test()` for one config file. We construct the env
    # afresh per task; the BrowserContext is rebuilt by env.reset.
    env = ScriptBrowserEnv(
        headless=not ns.render,
        slow_mo=ns.slow_mo,
        observation_type=ns.observation_type,
        current_viewport_only=ns.current_viewport_only,
        viewport_size={
            "width": ns.viewport_width,
            "height": ns.viewport_height,
        },
        save_trace_enabled=ns.save_trace_enabled,
        sleep_after_execution=ns.sleep_after_execution,
    )
    score = 0.0
    try:
        score = _run_one_task_loop(env, agent, ns, config_file)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return float(score)


def _run_one_task_loop(env, agent, args, config_file) -> float:
    """Inner loop: one task end-to-end. Mirrors the body of test() for one task."""
    render_helper = RenderHelper(config_file, args.result_dir, args.action_set_tag)
    try:
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            task_id = _c["task_id"]
            if _c["storage_state"]:
                cookie_file_name = os.path.basename(_c["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                auth_folder = os.environ.get("WEBARENA_AUTH_FOLDER", tempfile.mkdtemp())
                # Renew cookie in the per-worker auth folder.
                # Use sys.executable so the subprocess inherits this venv
                # (the system `python` lacks playwright). Prepend the project
                # root to PYTHONPATH so `from browser_env.env_config import …`
                # resolves when auto_login.py is invoked as a script.
                _project_root = Path(__file__).resolve().parent
                _env = {**os.environ}
                _pp = _env.get("PYTHONPATH", "")
                _env["PYTHONPATH"] = (
                    str(_project_root) + (os.pathsep + _pp if _pp else "")
                )
                # Clear any pre-existing stale cookie from a previous task so
                # a silent auto_login failure (which the audit flagged as a
                # high-severity stale-session risk) cannot succeed-with-bad-
                # data: the next assert will fire loudly instead.
                _expected_cookie = f"{auth_folder}/{cookie_file_name}"
                if os.path.exists(_expected_cookie):
                    try:
                        os.remove(_expected_cookie)
                    except OSError:
                        pass
                _login_proc = subprocess.run(
                    [
                        sys.executable,
                        str(_project_root / "browser_env" / "auto_login.py"),
                        "--auth_folder",
                        auth_folder,
                        "--site_list",
                        *comb,
                    ],
                    env=_env,
                    cwd=str(_project_root),
                )
                _c["storage_state"] = _expected_cookie
                assert os.path.exists(_c["storage_state"]), (
                    f"auto_login did not produce {_c['storage_state']} "
                    f"(returncode={_login_proc.returncode}, sites={comb})"
                )
                config_file_local = f"{auth_folder}/{os.path.basename(config_file)}"
                with open(config_file_local, "w") as f:
                    json.dump(_c, f)
                config_file = config_file_local

        logger.info(f"[Config file]: {config_file}")
        logger.info(f"[Intent]: {intent}")

        agent.reset(config_file)
        trajectory: Trajectory = []
        obs, info = env.reset(options={"config_file": config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory.append(state_info)

        meta_data = {"action_history": ["None"]}
        early_stop_thresholds = {
            "parsing_failure": args.parsing_failure_th,
            "repeating_action": args.repeating_action_failure_th,
        }
        while True:
            early_stop_flag, stop_info = early_stop(
                trajectory, args.max_steps, early_stop_thresholds
            )

            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                try:
                    action = agent.next_action(
                        trajectory, intent, meta_data=meta_data
                    )
                except ValueError as e:
                    action = create_stop_action(f"ERROR: {str(e)}")

            trajectory.append(action)

            action_str = get_action_description(
                action,
                state_info["info"]["observation_metadata"],
                action_set_tag=args.action_set_tag,
                prompt_constructor=agent.prompt_constructor
                if isinstance(agent, PromptAgent)
                else None,
            )
            render_helper.render(action, state_info, meta_data, args.render_screenshot)
            meta_data["action_history"].append(action_str)

            if action["action_type"] == ActionTypes.STOP:
                break

            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            if terminated:
                trajectory.append(create_stop_action(""))
                break

        evaluator = evaluator_router(config_file)
        score = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=env.page,
            client=env.get_page_client(env.page),
        )

        if args.save_trace_enabled:
            env.save_trace(
                Path(args.result_dir) / "traces" / f"{task_id}.zip"
            )
        return float(score)
    finally:
        try:
            render_helper.close()
        except Exception:
            pass


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie. Use sys.executable so the
                    # child inherits this venv's playwright install. Prepend
                    # the project root to PYTHONPATH so package imports work
                    # when auto_login.py is invoked as a script.
                    _project_root = Path(__file__).resolve().parent
                    _env = {**os.environ}
                    _pp = _env.get("PYTHONPATH", "")
                    _env["PYTHONPATH"] = (
                        str(_project_root) + (os.pathsep + _pp if _pp else "")
                    )
                    _login_proc = subprocess.run(
                        [
                            sys.executable,
                            str(_project_root / "browser_env" / "auto_login.py"),
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ],
                        env=_env,
                        cwd=str(_project_root),
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"]), (
                        f"auto_login did not produce {_c['storage_state']} "
                        f"(returncode={_login_proc.returncode}, sites={comb})"
                    )
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    # Apply per-worker env overrides BEFORE importing modules that read env at
    # import time. Some browser_env modules already imported above are env-
    # bound, but their lookup chain still re-reads os.environ for URL fields.
    _apply_worker_env(args)
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list)
