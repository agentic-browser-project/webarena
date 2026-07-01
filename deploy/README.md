# WebArena multi-worker deployment

Idempotent, parameterized scripts to stand up a **multi-worker WebArena
deployment** on one host: `N` isolated replicas of each mutable site
(shopping, shopping_admin, gitlab, reddit) on stride-spaced ports, plus the
shared read-only services (map, wikipedia, homepage) — all under a **rootless
Docker** daemon, with a world-readable socket so any local user can run reset.

Everything is driven by one file you edit per machine: **`config.env`**.

---

## 1. Architecture

Each site ships as **one pre-populated "golden" Docker image** (the official
WebArena images, dataset baked in). Replicas are *not* rebuilt per worker — the
same image is `docker run` `N` times, sharing layers copy-on-write, then each is
reconfigured to know its own URL:

```
golden image ──docker run×N──▶ replica wN on port (base + 100·N)
                                   ├─ rewrite base_url → http://HOST:port   (Magento setup:store-config:set / GitLab external_url)
                                   └─ snapshot datadir as <datadir>.golden   (→ ~25 s physical-swap reset)
```

- **Mutable sites** (per-worker, isolated DB+FS): created by `python -m mp.bring_up` (see [`../mp/`](../mp/)).
- **Shared services** (one instance, all workers point at it): map, wikipedia, homepage, each exposed on `0.0.0.0` via a `tcp_proxy.py` bridge.

---

## 2. Scripts

| File | Role |
|---|---|
| `config.env.example` | Copy to `config.env` and edit — the only per-machine change. |
| `lib.sh` | Shared helpers + config loader (sourced by all scripts). |
| `00_prereqs.sh` | **One-time, needs sudo**: rootless-docker packages, AppArmor userns fix, rootless install, reboot‑surviving `systemd --user` daemon on the big disk (0666 socket). |
| `setup_all.sh` | **Master**: 01→02→03→map→04→healthcheck. |
| `01_rootless_docker.sh` | Ensure the rootless daemon is up (no-op if `00_prereqs.sh` already started it). |
| `02_images.sh` | Obtain the 4 golden images + wiki `.zim` (`metis` \| `localtars` \| `peer`) and tag the `*-golden` aliases. |
| `03_shared_services.sh` | Wikipedia (kiwix) + homepage (Flask). |
| `deploy_map.sh` | Official WebArena-Verified map on `:13000` (data from `s3` or `peer`). |
| `04_bring_up_replicas.sh` | Write authoritative `config.json`, then `mp.bring_up` → all replicas + per-URL config + datadir snapshots + task configs. |
| `healthcheck.sh` | HTTP status matrix over every replica + shared service. |
| `tcp_proxy.py` | `0.0.0.0:<public> → 127.0.0.1:<internal>` bridge. |
| `fix_gitlab_puma.sh` | Recovery: cap puma + restart, for a GitLab replica that won't boot. |

---

## 3. Prerequisites

1. Linux host with a kernel supporting rootless containers. `00_prereqs.sh`
   installs the rest (`uidmap`, `fuse-overlayfs`, `slirp4netns`,
   `dbus-user-session`, `python3-venv`) and handles the **Ubuntu 23.10+/24.04
   AppArmor unprivileged-userns restriction** (otherwise rootlesskit fails with
   `fork/exec /proc/self/exe: permission denied`).
2. A **big disk** for `BASE` (~400–500 GB images + per-replica COW; +~140 GB if
   you deploy the map). If the host has no large shared mount, point all paths at
   a big local dir (e.g. `BASE=/home/$USER/webarena`).
3. The webarena **Python checkout** at `REPO` and a **venv** at `VENV`:
   ```bash
   python3 -m venv "$VENV"
   "$VENV/bin/pip" install -U pip -r "$REPO/requirements.txt" flask
   ```
   (Deployment itself — `mp.bring_up`/`mp.reset` — is stdlib-only; `flask` is for
   the homepage. The heavy agent deps in `requirements.txt` are not needed to
   *host* the sites and may fail to build without compilers — that's fine.)
4. Clone `web-arena-x/webarena-homepage` into `HOMEPAGE_DIR`.

---

## 4. Quick start

```bash
cd deploy
cp config.env.example config.env
$EDITOR config.env                 # set HOST_IP, NUM_WORKERS, paths, IMG_SOURCE

./00_prereqs.sh                    # one-time, sudo: rootless docker + AppArmor + systemd daemon

# get the data onto this host (see §6), then:
./setup_all.sh --skip_goldens     # images -> shared -> map -> bring up replicas -> healthcheck
```

All steps are idempotent — safe to re-run, or to add workers (bump
`NUM_WORKERS`, re-run `04_bring_up_replicas.sh`).

---

## 5. Port map

Port = `base + 100·worker`. `http://$HOST_IP:<port>/` (admin adds `/admin`).

| Site | base | w0 | w1 | … |
|---|---|---|---|---|
| shopping | 7770 | 7770 | 7870 | … |
| shopping_admin | 7780 | 7780 | 7880 | … |
| gitlab | 8023 | 8023 | 8123 | … |
| reddit | 9999 | 9999 | 10099 | … |

`8080`/`8085` are reserved (map internals), so **shopping_admin worker 3 shifts
7880→:8090** (mirrors `_HOST_PORT_RESERVED` in `mp/config.py`).
Shared: homepage `:4399`, wikipedia `:8888`, map `:13000`.

---

## 6. Getting the data onto a host

- **Images** (`02_images.sh`, `IMG_SOURCE`): `metis` (download the populated `.tar`s from the CMU mirror), `localtars` (load from `$BASE/downloads`), or `peer` (`docker save | docker load` over SSH from an already-deployed host). On a LAN, rsyncing the `.tar`s from a peer's `downloads/` then `IMG_SOURCE=localtars` is fastest.
- **Reset goldens** (`GOLDEN_ROOT`): either run bring-up **without** `--skip_goldens` to regenerate them locally from worker 0, **or** copy them from a peer:
  ```bash
  rsync -aH peer:/path/webarena-mp/golden/  "$GOLDEN_ROOT/"
  ```
  GitLab reset bind-mounts `$GOLDEN_ROOT/gitlab`, so it must be present.
- **Map data** (`deploy_map.sh`, `MAP_SOURCE`): `s3` (official bucket) or `peer` (stream the 9 volumes over the LAN).

---

## 7. Verify

```bash
./healthcheck.sh        # every replica + shared service; expect 200 (gitlab 302)
"$VENV/bin/python" -m mp.verify_golden --max_urls 20   # golden-equivalence (optional)
```

---

## 8. Operate

**Reset to pristine golden state** — per-site, usable by **any local user** (the
socket is `0666`, repo/venv world-readable), no SSH key needed:
```bash
DOCKER_HOST=unix://$BASE/rootless-docker/run/docker.sock \
PYTHONPATH=$REPO "$VENV/bin/python" -m mp.reset_cli \
    --site all --worker 0 --config "$REPO/mp/config.json"
```
Magento/Postmill reset is a ~25–110 s physical datadir swap; GitLab restores via
rsync from the golden tree, and if that stalls (common on `fuse-overlayfs`) the
built-in watchdog auto-recreates the replica from the golden image.

**After a reboot:** containers auto-start (`--restart unless-stopped`) and the
rootless daemon comes back via the `systemd --user` unit + linger. Re-run
`./03_shared_services.sh` to restart the host-side proxies/homepage, then
`./healthcheck.sh`.

---

## 9. Layout & notes

```
$BASE/rootless-docker/data   # image store (golden images, overlayfs)
$BASE/downloads              # site .tar images + wikipedia .zim
$BASE/{logs,pids}            # daemon/proxy/homepage logs + pidfiles
$(dirname $GOLDEN_ROOT)/     # webarena-mp state: golden/, auth/, config_files/, results/
```

- **Authoritative `config.json`:** `04_bring_up_replicas.sh` writes `mp/config.json` from `config.env` *before* bring-up (`lib.sh::write_config_json`). Don't rely on bring-up's own save — it runs last and doesn't derive `readonly_url_overrides`/state-paths from `--host`, so a copied config keeps the wrong host.
- **Hosts without a big shared mount:** set every path under a big local dir (e.g. `/home/$USER`); the scripts don't assume any specific mount.
- **Postmill 500 right after bring-up:** a transient postgres supervisor `startsecs` race under load — `docker exec forum[_wN] supervisorctl start postgres`.
- **Map disk size** varies by filesystem (ZFS compresses/keeps sparse; ext4 expands sparse files via `tar`) — same logical data either way.

---

## 10. Credentials (default WebArena test accounts)

| Site | Username | Password |
|---|---|---|
| shopping | `emma.lopez@gmail.com` | `Password.123` |
| shopping_admin (`/admin`) | `admin` | `admin1234` |
| gitlab | `byteblaze` | `hello1234` |
| reddit | `MarvelsGrantMan136` | `test1234` |
