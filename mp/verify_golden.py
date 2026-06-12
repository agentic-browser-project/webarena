"""Golden-equivalence verification.

For each URL an evaluator may navigate to:
  - GET it on a freshly reset worker_0 replica.
  - GET it on the ORIGINAL source container (still untouched).
  - Diff the responses after normalizing CSRF tokens, timestamps, nonces.

If any URL diverges, the reset primitive is buggy for that URL's site.

This script is the gate at the end of Phase 2 of the plan. It also produces a
"canary URL" list for cheap continuous verification during a real run.

Normalization removes a handful of well-known dynamic fields. New dynamic
fields uncovered during diff investigation should be added to NORMALIZERS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mp.config import MPConfig, load_config

log = logging.getLogger("mp.verify_golden")

# Dynamic fields that legitimately differ between two identical-state servers
# (CSRF tokens, timestamps, etc.). Each entry is (regex, replacement).
NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    # Magento form_key inputs
    (re.compile(r'name="form_key" value="[^"]+"'), 'name="form_key" value="X"'),
    # Magento CSRF in URLs
    (re.compile(r"/key/[a-f0-9]{32,}/?"), "/key/X/"),
    # GitLab CSRF meta tags
    (re.compile(r'<meta name="csrf-token" content="[^"]+"'), '<meta name="csrf-token" content="X"'),
    # GitLab CSP nonces
    (re.compile(r'nonce="[A-Za-z0-9+/=]+"'), 'nonce="X"'),
    # ISO-8601 timestamps inside data attributes
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"), "<TS>"),
    # GitLab "n seconds ago" / "n minutes ago"
    (re.compile(r"\b\d+ (?:second|minute|hour|day|week|month|year)s? ago\b"), "<RELTS>"),
    # Magento Bundled JS file hashes
    (re.compile(r"/version\d+/"), "/versionX/"),
    # Postmill: <time datetime="...">
    (re.compile(r'datetime="[^"]+"'), 'datetime="X"'),
    # Generic 32-hex hashes that appear in asset paths
    (re.compile(r"[a-f0-9]{32,64}\.(js|css|png|jpg|svg)"), r"ASSETHASH.\1"),
]


def normalize(body: str) -> str:
    for pat, rep in NORMALIZERS:
        body = pat.sub(rep, body)
    return body


def http_get(url: str, *, cookies: str = "", timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "webarena-mp-verify/1.0",
            **({"Cookie": cookies} if cookies else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, body
    except Exception:
        return 0, ""


@dataclass
class UrlDiff:
    url: str
    status_a: int
    status_b: int
    body_hash_a: str
    body_hash_b: str
    sample_a: str = ""
    sample_b: str = ""
    differs: bool = False


def extract_program_html_urls(repo_root: Path) -> list[str]:
    """Pull every concrete (non-'last', non-'func:') URL from program_html evals."""
    raw_path = repo_root / "config_files" / "test.raw.json"
    with raw_path.open() as f:
        raw = json.load(f)
    urls: set[str] = set()
    for task in raw:
        ev = task.get("eval", {})
        for ph in ev.get("program_html", []) or []:
            url = ph.get("url") or ""
            if not url:
                continue
            if url == "last" or url.startswith("func:") or "__" in url and url.startswith("func:"):
                continue
            if url.startswith("__") and url.endswith("__"):
                continue
            if url.startswith("func"):
                continue
            urls.add(url)
        for ref in (ev.get("reference_url") or "").split(" |OR| "):
            ref = ref.strip()
            if ref and "://" in ref:
                urls.add(ref)
    return sorted(urls)


def substitute_urls(urls: list[str], cfg: MPConfig, worker_id: int) -> list[str]:
    """Replace __SHOPPING__/__GITLAB__/... placeholders with worker_id URLs."""
    env = cfg.env_for(worker_id)
    substituted = []
    for u in urls:
        out = u
        for k, v in env.items():
            out = out.replace(f"__{k}__", v)
        substituted.append(out)
    return substituted


def verify_pair(url_worker: str, url_source: str) -> UrlDiff:
    sa, ba = http_get(url_worker)
    sb, bb = http_get(url_source)
    na = normalize(ba)
    nb = normalize(bb)
    ha = hashlib.sha256(na.encode("utf-8")).hexdigest()[:16]
    hb = hashlib.sha256(nb.encode("utf-8")).hexdigest()[:16]
    diff = (sa != sb) or (ha != hb)
    sample_a = ""
    sample_b = ""
    if diff:
        # Find first divergence
        i = 0
        n = min(len(na), len(nb))
        while i < n and na[i] == nb[i]:
            i += 1
        sample_a = na[max(0, i - 50): i + 200]
        sample_b = nb[max(0, i - 50): i + 200]
    return UrlDiff(
        url=url_worker,
        status_a=sa,
        status_b=sb,
        body_hash_a=ha,
        body_hash_b=hb,
        sample_a=sample_a,
        sample_b=sample_b,
        differs=diff,
    )


def run_verification(
    cfg: MPConfig,
    *,
    source_urls: dict[str, str],
    worker_id: int = 0,
    max_urls: int | None = None,
) -> dict[str, list[UrlDiff]]:
    """Compare worker_0's responses to the source container's.

    ``source_urls`` maps {site_key -> base URL of an untouched container}.
    We substitute __SHOPPING__/__GITLAB__/__REDDIT__/__SHOPPING_ADMIN__/__MAP__/__WIKIPEDIA__
    in `extract_program_html_urls` with worker_0 URLs (for "test under test") and
    with source URLs (for "ground truth"), then diff.
    """
    repo_root = Path(__file__).resolve().parent.parent
    all_urls = extract_program_html_urls(repo_root)
    if max_urls:
        all_urls = all_urls[:max_urls]

    # Substitute for worker
    worker_urls = substitute_urls(all_urls, cfg, worker_id)

    # Substitute for source (build an MPConfig clone whose URLs are source URLs)
    import dataclasses
    source_overrides = {
        "wikipedia": source_urls.get("WIKIPEDIA", cfg.url_for("wikipedia", worker_id)),
        "map": source_urls.get("MAP", cfg.url_for("map", worker_id)),
        "homepage": source_urls.get("HOMEPAGE", cfg.url_for("homepage", worker_id)),
    }
    # We need to also override mutable site URLs. Easiest: monkey-patch a copy.
    source_cfg = dataclasses.replace(cfg, readonly_url_overrides=source_overrides)

    # For mutable, replace url_for behaviour via the substituted strings directly.
    def sub_source(u: str) -> str:
        out = u
        for k, v in source_urls.items():
            out = out.replace(f"__{k}__", v)
        return out
    source_substituted = [sub_source(u) for u in all_urls]

    diffs_by_site: dict[str, list[UrlDiff]] = defaultdict(list)
    for u_w, u_s, orig in zip(worker_urls, source_substituted, all_urls):
        # Identify site key from the original URL.
        site_key = "unknown"
        for k in ("SHOPPING_ADMIN", "SHOPPING", "GITLAB", "REDDIT", "WIKIPEDIA", "MAP", "HOMEPAGE"):
            if f"__{k}__" in orig:
                site_key = k.lower()
                break
        diff = verify_pair(u_w, u_s)
        if diff.differs:
            log.warning(
                "DIFFERS [%s] %s status %d vs %d hash %s vs %s",
                site_key, u_w, diff.status_a, diff.status_b, diff.body_hash_a, diff.body_hash_b,
            )
        diffs_by_site[site_key].append(diff)
    return diffs_by_site


def _argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify golden-equivalence")
    p.add_argument("--max_urls", type=int, default=None, help="cap for fast smoke tests")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--source_shopping",
        default=None,
        help="URL of the unmolested source shopping (defaults to base port 7770)",
    )
    p.add_argument("--source_shopping_admin", default=None)
    p.add_argument("--source_reddit", default=None)
    p.add_argument("--source_gitlab", default=None)
    p.add_argument("--source_map", default=None)
    p.add_argument("--source_wikipedia", default=None)
    p.add_argument("--source_homepage", default=None)
    p.add_argument("--out", default=None, help="write JSON report to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparse().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    # Defaults: source = base ports
    sources = {
        "SHOPPING": args.source_shopping or f"http://{cfg.host}:7770",
        "SHOPPING_ADMIN": args.source_shopping_admin or f"http://{cfg.host}:7780/admin",
        "REDDIT": args.source_reddit or f"http://{cfg.host}:9999",
        "GITLAB": args.source_gitlab or f"http://{cfg.host}:8023",
        "MAP": args.source_map or f"http://{cfg.host}:13000",
        "WIKIPEDIA": args.source_wikipedia
        or f"http://{cfg.host}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": args.source_homepage or f"http://{cfg.host}:4399",
    }
    diffs = run_verification(cfg, source_urls=sources, worker_id=0, max_urls=args.max_urls)
    summary: dict[str, dict[str, int]] = {}
    for site, ds in diffs.items():
        summary[site] = {
            "total": len(ds),
            "differs": sum(1 for d in ds if d.differs),
        }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "diffs": [
                        {k: v for k, v in vars(d).items() if k != "sample_a" or d.differs}
                        for ds in diffs.values()
                        for d in ds
                    ],
                },
                indent=2,
            )
        )
    any_diff = any(s["differs"] > 0 for s in summary.values())
    return 1 if any_diff else 0


if __name__ == "__main__":
    sys.exit(main())
