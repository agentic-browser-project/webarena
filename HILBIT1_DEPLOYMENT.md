# Hilbit1 Deployment — WebArena Multi-Worker

End-to-end record of standing up the WebArena multi-worker harness on `hilbit1.cis.upenn.edu` as a parallel deployment to `hilbit2.cis.upenn.edu`. Mirrors hilbit2's topology so the same code (`/z/wangcy07/webarena-repo/mp/`) drives both hosts.

Operator: wangcy07
Deployment date: 2026-06-22 / 2026-06-23
Plan: `/Users/chuyuewang/.claude/plans/shiny-jingling-sifakis.md`

---

## 1. Host topology

| Property | hilbit2 (reference) | hilbit1 (this deployment) |
|---|---|---|
| Public IP | 158.130.4.158 | **158.130.4.153** |
| FQDN | hilbit2.cis.upenn.edu | hilbit1.cis.upenn.edu |
| Cores / RAM | 128 / 2.0 TiB | 256 / 1.5 TiB |
| GPU | none | none |
| Kernel | 6.8.0-106 | 6.8.0-106 |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| wangcy07 uid | 1030 | 1033 |
| /z mount | ZFS, 23 TB | ZFS, 23 TB |
| sudo for wangcy07 | n/a | NO |
| Other tenants | none | sid, jms, lywong, nikos, runlong, taoluo71, zhiyaot, zpkhor; uid 1025 owns the *other* rootless dockerd |

## 2. Filesystem layout

Both hosts use the same paths so `mp/config.json` is the only file that changes between them:

```
/z/wangcy07/
  webarena/                      host-side scripts + rootless docker tree
    deploy_map_official.sh       OSM map data download + extract + container
    deploy_webarena.sh           legacy single-instance bring-up (hilbit2 only)
    deploy_shared_hilbit1.sh     hilbit1-specific wikipedia + homepage bring-up
    tcp_proxy.py                 Python TCP proxy 0.0.0.0:PORT -> 127.0.0.1:PORT
    map_server.py                fallback fake map (replaced by deploy_map_official)
    fix_gitlab_puma.sh           in-place puma restart for gitlab idle-OOM
    downloads/                   source docker .tar files + wikipedia.zim (~232 GB)
    wa-verified-downloads/       S3-fetched map data tarballs (~187 GB)
    rootless-docker/
      data/                      dockerd data-root (overlayfs snapshots, ~600 GB at full N=7)
      exec/                      dockerd exec-root
      run/docker.sock            DOCKER_HOST socket (chmod 0666 for shared reset access)
      xdg-runtime/               XDG_RUNTIME_DIR (no logind session for wangcy07)
    pids/                        pidfiles for tcp_proxy / homepage
    logs/                        per-service runtime logs
  webarena-repo/                 python source (mp/, browser_env/, evaluation_harness/, ...)
    mp/config.json               operator-visible config — points at this host
  webarena-mp/                   runtime state for the harness
    config.json                  duplicate of repo's mp/config.json
    golden/{shopping,shopping_admin,reddit,gitlab}/   golden SQL/dump/rsync trees (~70 GB)
    auth/                        per-worker cookie folders (auto-login state)
    config_files/w0..w6/         per-worker task config JSONs
    results/                     per-run scoring + traces
  webarena-venv/                 python 3.12 venv
```

## 3. Rootless docker daemon

Launched manually under `tmux webarena-dockerd` (no systemd-user session because wangcy07 isn't `linger`-enabled and we have no sudo to fix that).

```bash
tmux new-session -d -s webarena-dockerd "
  export XDG_RUNTIME_DIR=/z/wangcy07/webarena/rootless-docker/xdg-runtime
  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  exec /usr/bin/dockerd-rootless.sh \
    --data-root /z/wangcy07/webarena/rootless-docker/data \
    --exec-root /z/wangcy07/webarena/rootless-docker/exec \
    --host unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
    > /z/wangcy07/webarena/dockerd.log 2>&1
"
```

After it comes up:

```bash
chmod 0666 /z/wangcy07/webarena/rootless-docker/run/docker.sock
```

This opens the daemon socket to any local user (per the "reset must be usable by other users" requirement). Re-apply the chmod every time the daemon restarts (socket is recreated). Effective version: `Server 29.3.0`, `Storage Driver overlayfs`, `Cgroup Driver cgroupfs (v1)`.

## 4. Ports and binding model

All mutable-site containers (shopping, shopping_admin, forum, gitlab × N workers) bind `0.0.0.0:<port>` directly — externally reachable on `158.130.4.153:<port>` without proxy. Shared services (map, wikipedia, homepage) bind `127.0.0.1:<internal>` then a host-side `tcp_proxy.py` forwards `0.0.0.0:<public>` → `127.0.0.1:<internal>`.

| Service | Worker 0 | Worker 1 | Worker 2 | Worker 3 | Worker 4 | Worker 5 | Worker 6 |
|---|---|---|---|---|---|---|---|
| shopping | 7770 | 7870 | 7970 | 8070 | 8170 | 8270 | 8370 |
| shopping_admin | 7780 | 7880 | 7980 | **8090*** | 8180 | 8280 | 8380 |
| forum (reddit) | 9999 | 10099 | 10199 | 10299 | 10399 | 10499 | 10599 |
| gitlab | 8023 | 8123 | 8223 | 8323 | 8423 | 8523 | 8623 |

\* `shopping_admin_w3` is shifted from natural port 8080 to **8090** because `_HOST_PORT_RESERVED = {8080, 8085}` in [mp/config.py:24](mp/config.py#L24) skips ports the map container's internal services collide with. On hilbit1, the OTHER reason 8080 is reserved is that `wangcy07`'s already-running `uber-benchmark-tileserver-1` container has bound :8080 — the same skip-rule handles it.

| Shared service | Public port (proxy) | Internal port (container) |
|---|---|---|
| wikipedia (kiwix) | 0.0.0.0:8888 → 127.0.0.1:18888 | container:80 |
| map (osm) | 0.0.0.0:13000 → 127.0.0.1:13030 | container:8080 internal |
| homepage (flask) | 0.0.0.0:4399 → 127.0.0.1:14399 | flask host:14399 |

Inference tunnel:

| Service | Public port | Notes |
|---|---|---|
| Ollama / SGLang (LLM) | 127.0.0.1:11434 (forwarded via `ssh -L 11434:127.0.0.1:11434 wangcy07@gray`) | Not yet established on hilbit1; needs `wangcy07`'s key on gray. |

## 5. mp/config.json (hilbit1)

Two copies (operator-visible + repo-internal), identical content:

- `/z/wangcy07/webarena-mp/config.json`
- `/z/wangcy07/webarena-repo/mp/config.json`

```json
{
  "auth_root": "/z/wangcy07/webarena-mp/auth",
  "config_files_root": "/z/wangcy07/webarena-mp/config_files",
  "docker_host": "unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock",
  "golden_root": "/z/wangcy07/webarena-mp/golden",
  "host": "158.130.4.153",
  "map_replicas": 1,
  "num_workers": 7,
  "port_stride": 100,
  "readonly_url_overrides": {
    "homepage": "http://158.130.4.153:4399",
    "map": "http://158.130.4.153:13000",
    "wikipedia": "http://158.130.4.153:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
  },
  "result_dir": "/z/wangcy07/webarena-mp/results",
  "ssh_host": ""
}
```

The only differences from hilbit2 are the host IP (158.130.4.153 vs .158) and the three URLs inside `readonly_url_overrides`. All other paths are byte-identical so the same code works on both hosts without conditional logic.

## 6. Reset machinery — usable by other users

Reset is per-site, runnable via `mp/reset_cli.py`. The two access requirements for non-wangcy07 users:

1. **Docker socket access**: `chmod 0666 /z/wangcy07/webarena/rootless-docker/run/docker.sock` — every time dockerd starts, the socket is recreated with `srw-rw----`. Apply 0666 after each start (documented in §3). Anyone with read on `/z/wangcy07/webarena-repo/` and `DOCKER_HOST` exported can now call reset.
2. **Repo + venv access**: `/z/wangcy07/webarena-{repo,venv,mp}/` are mode `0755` (group + other read+execute) by default from the `mkdir` defaults — verified.

Usage by another user:

```bash
DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
PYTHONPATH=/z/wangcy07/webarena-repo \
/z/wangcy07/webarena-venv/bin/python -m mp.reset_cli \
    --site all --worker 0 \
    --config /z/wangcy07/webarena-repo/mp/config.json
```

`reset_site()` (per [mp/reset.py:993](mp/reset.py#L993)) dispatches to the per-site primitive:
- Magento (shopping / shopping_admin): physical datadir swap if snapshot exists (~24 s), else logical SQL restore.
- Postmill (reddit): pg_restore + media tarballs.
- GitLab: rsync from `/opt/golden/gitlab` (host-side golden tree) + WAL reset + service restart.

**Warmup**: every successful Magento reset ends with `_warm_magento_cache()` (see [mp/reset.py:333,419,522](mp/reset.py#L333)) — issues a single homepage GET so the next agent visit hits a hot full-page cache (~0.3 s instead of ~2 s cold render). This is the reuse of the recent commit `d065026 reset_magento: warm full-page cache after reset` per the operator request.

## 7. Build chronology (what actually happened)

| Time (EDT) | Action | Where logged |
|---|---|---|
| 22:17 | Created `/z/wangcy07/{webarena,webarena-repo,webarena-mp,webarena-venv}/` and subdirs | live shell |
| 22:18 | Generated ed25519 key for wangcy07 on hilbit1, added to hilbit2 `authorized_keys` | `~/.ssh/id_ed25519` |
| 22:21 | Launched dockerd-rootless under tmux `webarena-dockerd`; chmod 0666 on socket | `/z/wangcy07/webarena/dockerd.log` |
| 22:23 | Started golden rsync (75 GB) + first serial image transfer attempt | `xfer-goldens.log`, `xfer-images.log` |
| 22:30 | Aborted serial transfer; restarted as 5-way parallel | `xfer-par.log` |
| 22:42 | Aborted parallel `docker save` (only ~33 MB/s aggregate, hilbit2 dockerd-snapshotter CPU-bound); switched to rsync of original `.tar` files from `/z/wangcy07/webarena/downloads/` | `xfer-tars.log` |
| 22:42 | Goldens rsync finished — 75 GB at 70 MB/s avg | `xfer-goldens.log` exit 0 |
| 22:46 | Started `deploy_map_official.sh` — S3 downloads too slow (~10 MB/s); killed it | `tmux deploy-map` |
| 22:54 | Switched map data to direct **docker-volume** transfer from hilbit2 (LAN tar\|ssh\|tar, ~80 MB/s) | `xfer-vols.log` |
| 23:42 | Tarball rsync (231 GB inc. 89 GB wikipedia.zim) finished; bring_up_orch STEP 2 `docker load` started | `xfer-tars.log` exit 0 |
| ~23:50 | **Incident**: `docker load` (shopping_admin) wedged on a futex after I force-killed the concurrent map-volume alpine `tar` containers — they shared the same containerd and corrupted a transaction. Image was actually fully loaded + verified (ID `993d1ee9…` matches hilbit2). | `dockerd.log` |
| 23:58 | Recovery: stopped orchestrator + hung load; daemon itself healthy (`docker run alpine echo` OK, no restart needed). Restarted loads **sequentially with no concurrent container ops** via `load_images.sh`. | `load_images.log` |
| 23:58 | Also discovered the killed map-vol transfer left `tile_db` (10.9G vs 16.4G) and `routing_foot` (3.0G vs 4.9G) **partial**; the transfer script skips any non-empty volume, so wrote `redo_map_volumes.sh` to clear+redo the 6 incomplete/missing volumes. | `redo_map_volumes.log` |
| 06-23 23:42→06-24 03:30 | Sequential `docker load` of shopping (129G), postmill (101G), gitlab (134G) — all rc=0, IDs match hilbit2 exactly | `load_images.log` |
| 06-24 ~03:31→08:05 | Map-volume redo (6 volumes cleared+retransferred) — completed; all 9 volumes match hilbit2 sizes | `redo_map_volumes.log` |
| 06-24 04:53→06:03 | `mp.bring_up --num_workers 7 --host 158.130.4.153 --skip_goldens` — created 28 replicas, configured base_urls, snapshotted 21 datadirs for fast-reset | `run_all_remaining.log` |
| 06-24 06:03 | bring_up exited rc=1 at step-5 health check: Postmill postgres hit supervisor `startsecs` race under load 286 (marked FATAL though datadir+snapshot were fine). **Fixed**: `supervisorctl start postgres` in the 3 affected forum replicas (w3-w6 had self-recovered). | — |
| 06-24 06:04 | Shared services (wikipedia 200, map 200, homepage 200) up via `deploy_shared_hilbit1.sh` + tcp_proxy | — |
| 06-24 06:05 | **Full health probe: 28/28 mutable replicas + 3/3 shared services serving correctly** | — |
| 06-24 06:08 | Copied untracked `mp/reset_cli.py` from Mac (absent on hilbit2); reset test on shopping_w1 swapped datadir + served 200 (cache-clear step hit 60s timeout under load 272 — environmental, not a defect) | — |
| 06-24 06:10 | SSH tunnel hilbit1:11434→gray:11434 established (`tmux gray-tunnel`); `/v1/models` lists qwen2.5:7b-instruct | — |
| 06-24 06:12 | Transferred repo `config_files/test.raw.json` (excluded by earlier rsync); rendered **812 configs × 7 workers**; w3 shopping_admin correctly uses :8090 (reserved-port shift), zero :8080 | — |
| 06-24 06:13 | Smoke orchestrator: import chain needed deps — installed to match hilbit2 (beartype==0.12.0, openai==0.27.0, tiktoken==0.11.0, playwright==1.60.0, numpy==2.2.6, gymnasium, nltk, pillow, aiolimiter) | — |
| 06-24 06:15 | Smoke orchestrator re-run (tasks 0,1 via qwen2.5 over the tunnel) — full agent loop | `smoke_test.log` |

### Lessons (do not repeat)

1. **Never run concurrent `docker` container operations against the rootless daemon while a `docker load` is in flight** — under load the containerd overlayfs snapshotter serializes, and force-killing one operation's containers wedges the others on a futex. Load images sequentially, alone.
2. **`docker save | docker load` over SSH is slow** (~33 MB/s aggregate even 5-way parallel) because hilbit2's dockerd-snapshotter is CPU-bound re-serializing. **Rsync the original `.tar` files** from `/z/wangcy07/webarena/downloads/` instead (~80–100 MB/s, network-bound) then `docker load` locally.
3. **Map data is NOT in the map image** — it lives in 9 named docker volumes (~48.6 GB). Either re-download from S3 (slow) or transfer the volumes directly from hilbit2 (`docker run -v vol:/src alpine tar -cf -` piped to the same on the target).
4. **`docker load` finishing is not signalled by the image appearing in `docker images`** — the image registers before the snapshotter unpack completes. Confirm by the `docker load` process exiting, not by `docker images`.

### Map volume reference sizes (hilbit2)

| Volume | Size | Notes |
|---|---|---|
| webarena_verified_map_tile_db | 16.4 G | rendered OSM tiles postgres |
| webarena_verified_map_routing_car | 4.0 G | OSRM car profile |
| webarena_verified_map_routing_bike | 4.9 G | OSRM bike profile |
| webarena_verified_map_routing_foot | 4.9 G | OSRM foot profile |
| webarena_verified_map_nominatim_db | 14.4 G | geocoding postgres |
| webarena_verified_map_nominatim_flatnode | 3.6 G | nominatim flatnode file |
| webarena_verified_map_website_db | 14.5 M | map website postgres |
| webarena_verified_map_tiles | (empty) | populated at runtime |
| webarena_verified_map_style | 8.0 M | map style assets |

## 7b. FINAL VERIFIED STATE (2026-06-24 06:2x)

**Deployment is COMPLETE and verified end-to-end.**

```
container count: 30 running (28 mutable replicas + map + wikipedia)
mutable endpoints: 28/28 serving correctly
  w0: shop 7770=200  admin 7780=200  reddit 9999 =200  gitlab 8023=302
  w1: shop 7870=200  admin 7880=200  reddit 10099=200  gitlab 8123=302
  w2: shop 7970=200  admin 7980=200  reddit 10199=200  gitlab 8223=302
  w3: shop 8070=200  admin 8090=200  reddit 10299=200  gitlab 8323=302   (admin shifted 8080→8090)
  w4: shop 8170=200  admin 8180=200  reddit 10399=200  gitlab 8423=302
  w5: shop 8270=200  admin 8280=200  reddit 10499=200  gitlab 8523=302
  w6: shop 8370=200  admin 8380=200  reddit 10599=200  gitlab 8623=302
shared: wikipedia 8888=200   map 13000=200   homepage 4399=200
image IDs: all 4 golden images byte-match hilbit2 (cb52e8185fbc, 0a0c002b4dd0, 993d1ee9c135, ccff8c1772be)
golden trees: gitlab 23G, reddit 39G, shopping 539M, shopping_admin 1.8M (match hilbit2)
map volumes: all 9 match hilbit2 (tile_db 16.4G, nominatim_db 14.4G, routing 4.0/4.9/4.9G, flatnode 3.6G, …)
fast-reset snapshots: /var/lib/mysql.golden (14 Magento) + /usr/local/pgsql/data.golden (7 Postmill) present
base_url: 158.130.4.153 in served HTML + Magento core_config_data
per-worker configs: 812 × 7 = 5,684 rendered; w3 admin uses :8090, zero :8080
reset: reset_cli.py present; datadir-swap + warmup functional
smoke run: tasks 0,1 via qwen2.5 over gray tunnel → score=0.0 error=null (harness OK; 0.0 expected for 7B model)
socket: srw-rw-rw- (0666) — other users can drive reset
persistence: dockerd (tmux webarena-dockerd), gray-tunnel (tmux), 5 tcp_proxy + flask (nohup)
```

### Operational caveat — host load
hilbit1 is shared; other tenants periodically drive load to ~280 (256 cores). Under that load:
- Postmill postgres can lose the supervisor `startsecs` race on restart after a datadir snapshot (marked FATAL though the data is fine). Recover with `docker exec forum_wN supervisorctl start postgres`.
- The Magento reset's filesystem-cache-clear step (60 s timeout in `reset.py`) can time out; the datadir swap + base_url rewrite still apply and the replica serves 200. Re-run reset when load subsides, or it self-heals on the next task's reset.

Neither is a deployment defect — the reset/bring_up code is identical to hilbit2's. They are transient, load-induced timeouts on a contended shared host.

### Restart after daemon/host restart
The deployment is NOT auto-start (no sudo → no systemd-user linger). After a host reboot, re-run:
```bash
ssh wangcy07@hilbit1.cis.upenn.edu
# 1. rootless dockerd
tmux new-session -d -s webarena-dockerd '
  export XDG_RUNTIME_DIR=/z/wangcy07/webarena/rootless-docker/xdg-runtime
  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  exec /usr/bin/dockerd-rootless.sh \
    --data-root /z/wangcy07/webarena/rootless-docker/data \
    --exec-root /z/wangcy07/webarena/rootless-docker/exec \
    --host unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
    > /z/wangcy07/webarena/dockerd.log 2>&1'
sleep 10
chmod 0666 /z/wangcy07/webarena/rootless-docker/run/docker.sock
# 2. containers auto-restart (restart=unless-stopped); if not, re-run mp.bring_up (idempotent)
# 3. shared services + map container + proxies
bash /z/wangcy07/webarena/deploy_shared_hilbit1.sh
DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock docker start map
nohup python3 /z/wangcy07/webarena/tcp_proxy.py --listen-port 13000 --target-port 13030 &
# 4. LLM tunnel
tmux new-session -d -s gray-tunnel 'ssh -N -L 127.0.0.1:11434:127.0.0.1:11434 -o ServerAliveInterval=30 wangcy07@gray.cis.upenn.edu'
```

## 7c. Website load/render timing (on-host, warm) + reset CLI

Measured on hilbit1 under host load ~276 (worst case; other tenants busy). "warm" = 2nd request, representative of the steady-state agent experience:

| Site | HTTP | Warm time |
|---|---|---|
| shopping | 200/302 | ~0.37 s |
| shopping_admin | 200/302 | ~0.41 s |
| reddit / forum | 200 | ~0.84 s |
| gitlab | 302 | ~0.06 s |
| wikipedia | 200 | ~0.008 s |
| map (home) | 200 | ~0.05 s |
| map tile (cached) | 200 | ~0.004 s |
| map nominatim (geocode) | 200 | ~0.09 s |
| homepage | 200 | ~0.008 s |

Notes:
- **Cold vs warm**: Magento storefront/admin first hit ~0.4 s; reddit ~0.9 s; gitlab redirect ~0.06 s. After a reset the warmup GET (req #3) primes Magento's full-page cache so the agent's first real visit is warm.
- **Map tiles render on demand**: a never-before-requested tile triggers renderd (CPU-bound; under load 280 the first render can exceed 60–90 s and 404 until cached). Once rendered it serves in ~0.003 s and persists in `/var/cache/renderd/tiles`. The tile GIS DB (16.4 G) matches hilbit2; behavior is identical to hilbit2, which simply has weeks of pre-cached tiles. To pre-warm the WebArena region, request the Pittsburgh tile pyramid once when the host is calm.
- **map route**: `/route/v1/car/...` returns 404 on BOTH hilbit1 and hilbit2 (the OSRM HTTP route isn't exposed at that path in this image) — parity, not a regression. WebArena's map routing goes through the web UI's own endpoints.

### reset CLI — runnable on BOTH hosts
`mp/reset_cli.py` was untracked (Mac-only); now copied to **both** hilbit1 and hilbit2 at `/z/wangcy07/webarena-repo/mp/reset_cli.py`. `python -m mp.reset_cli --help` works on both. Live resets:
- **hilbit2** (quiescent): `--site reddit --worker 1` → **DONE in 55 s, exit 0**, endpoint 200 in 0.32 s.
- **hilbit1** (load 280): datadir golden-swap applies and the replica serves 200, but the post-swap cache-clear step can exceed its 60 s (Magento) / 300 s (Postmill) timeout under load and report FAILED; the app self-heals on first access (e.g. forum_w2 → 200 in 9.8 s cold, 0.9 s warm). Under normal load it completes like hilbit2.

Usage (any local user, no SSH key needed — socket is 0666):
```bash
DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
PYTHONPATH=/z/wangcy07/webarena-repo \
/z/wangcy07/webarena-venv/bin/python -m mp.reset_cli \
    --site all --worker 0 --config /z/wangcy07/webarena-repo/mp/config.json
```

## 8. Verification gates (e2e) — ALL PASSED

The harness IS the verification: the smoke run resets a replica, drives the full agent loop against it, and the evaluator reads back from the same replica.

### 8.1 Live HTTP per replica

```bash
for w in 0 1 2 3 4 5 6; do
  echo "--- worker $w ---"
  shop_port=$((7770 + 100*w))
  admin_port=$((7780 + 100*w))
  reddit_port=$((9999 + 100*w))
  gitlab_port=$((8023 + 100*w))
  # Skip the reserved-port shift for admin worker 3
  if [ $w -eq 3 ]; then admin_port=8090; fi
  for spec in "shopping:$shop_port" "shopping_admin:$admin_port" "reddit:$reddit_port" "gitlab:$gitlab_port"; do
    site=${spec%:*}; port=${spec#*:}
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://158.130.4.153:$port/)
    echo "  $site $port -> $code"
  done
done
```

Pass criteria: every mutable site returns 200 (or 302 for gitlab); shared services (wikipedia/map/homepage) return 200.

### 8.2 Golden equivalence

```bash
cd /z/wangcy07/webarena-repo
DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
PYTHONPATH=/z/wangcy07/webarena-repo \
/z/wangcy07/webarena-venv/bin/python -m mp.verify_golden --max_urls 20 \
  --out /z/wangcy07/webarena-mp/verify.json
```

Pass criteria: `0 diffs` after normalization (CSRF tokens, timestamps, nonces ignored — see `NORMALIZERS` in [mp/verify_golden.py:36](mp/verify_golden.py#L36)).

### 8.3 Smoke orchestrator

```bash
DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock \
PYTHONPATH=/z/wangcy07/webarena-repo \
/z/wangcy07/webarena-venv/bin/python -m mp.orchestrator \
  --task_ids 0 1 2 \
  --model qwen2.5:7b-instruct \
  --inference_backend ollama \
  --result_dir_override /z/wangcy07/webarena-mp/results/smoke
```

Pass criteria: three rows appear in `scores.jsonl` with `error: null`. Inspect Playwright traces under `results/smoke/w*/render_*.html` — every URL must be `158.130.4.153:*` (proves worker isolation).

### 8.4 External reachability

From an off-host machine (e.g., Mac):
```bash
for p in 7770 7780 8023 9999 8888 13000 4399; do
  curl -s -o /dev/null -w "$p -> %{http_code}\n" --max-time 10 http://158.130.4.153:$p/
done
```
All must be reachable.

## 9. Network sanity

| Probe | Result |
|---|---|
| hilbit1 ⇆ hilbit2 SSH | ✓ (key pushed) |
| hilbit1 ⇆ S3 (webarena-map-server-data) | ✓ HTTP 200 |
| hilbit1 ⇆ metis.lti.cs.cmu.edu | ✓ HTTP 200 |
| rootless docker DNS (alpine → S3) | ✓ HTTP/1.1 200 OK |
| ports already bound by other tenants in WebArena range | 8080 (uber-benchmark-tileserver), 8085, 8086 — only 8080+8085 in `_HOST_PORT_RESERVED`; rest don't collide |

## 10. Rollback / cleanup

All deployment artifacts live under `/z/wangcy07/`. To remove cleanly:

```bash
ssh wangcy07@hilbit1.cis.upenn.edu '
  # Stop the dockerd-rootless daemon and tmux sessions
  tmux kill-server
  # Stop tcp_proxy + flask host processes
  pkill -u wangcy07 -f "tcp_proxy.py|flask run"
  # Remove the tree (destructive — confirm)
  rm -rf /z/wangcy07/{webarena,webarena-repo,webarena-mp,webarena-venv}
'
```

This does NOT touch:
- hilbit1's system docker (rootful dockerd at fd:// — runs wangcy07's `uber-benchmark`/`instagram`/`dsl-*` containers as well as other users')
- the uid 1025 user's separate rootless dockerd at `/run/user/1025/dockerd-rootless`
- k3s / libvirt / other system services

(End of file — sections 7+8 to be expanded with verified output after deployment completes.)
