"""Multi-worker WebArena benchmark runner.

Module layout:
    config        — runtime config: N workers, ports, paths, hosts
    reset         — per-site reset primitives (§4, §14 of plan)
    bring_up      — replica + golden creation (§7.2)
    worker        — per-worker task loop (§7.7)
    orchestrator  — task queue + worker supervisor (§7.5, §9)
    verify_golden — byte-equivalence verification (§8)
"""

from mp.config import MPConfig, load_config

__all__ = ["MPConfig", "load_config"]
