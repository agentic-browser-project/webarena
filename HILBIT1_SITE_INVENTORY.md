# Hilbit1 WebArena Site Inventory

Deployment record for the WebArena multi-worker deployment on **hilbit1.cis.upenn.edu (158.130.4.153)**, N=7 workers. Same column format as the baseline tracking sheet. Companion to [HILBIT1_DEPLOYMENT.md](HILBIT1_DEPLOYMENT.md).

Recorded: 2026-06-24 · Analyzer: chuyue · Images: byte-identical to hilbit2 (same docker image IDs).

## Site table

| Machine name | URL:Port | Port | Absolute address of docker files | Baseline Repo | Web Information | Analyzer | PHP / runtime | Open to campus network? | Scanned? | Dynamic code? | Dynamic action captured? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hilbit1.cis.upenn.edu | http://158.130.4.153:7770 | 7770 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w0) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:7870 | 7870 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w1) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:7970 | 7970 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w2) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8070 | 8070 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w3) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8170 | 8170 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w4) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8270 | 8270 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w5) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8370 | 8370 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping (w6) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:7780/admin | 7780 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w0) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:7880/admin | 7880 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w1) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:7980/admin | 7980 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w2) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8090/admin | 8090 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w3) — note: shifted from 8080 (reserved) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8180/admin | 8180 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w4) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8280/admin | 8280 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w5) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8380/admin | 8380 | /z/wangcy07/webarena/rootless-docker/data | WebArena | shopping_admin (w6) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:9999 | 9999 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w0) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10099 | 10099 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w1) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10199 | 10199 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w2) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10299 | 10299 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w3) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10399 | 10399 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w4) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10499 | 10499 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w5) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:10599 | 10599 | /z/wangcy07/webarena/rootless-docker/data | WebArena | reddit/forum (w6) | chuyue | PHP 8.1.17 (no upgrade) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8023 | 8023 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w0) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8123 | 8123 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w1) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8223 | 8223 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w2) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8323 | 8323 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w3) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8423 | 8423 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w4) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8523 | 8523 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w5) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8623 | 8623 | /z/wangcy07/webarena/rootless-docker/data | WebArena | gitlab (w6) | chuyue | N/A (Ruby/Rails) | Y | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing | 8888 | /z/wangcy07/webarena/downloads/wikipedia_en_all_maxi_2022-05.zim | WebArena | wikipedia (shared, read-only) | chuyue | N/A (C++ binary) | Y (proxy 0.0.0.0:8888 → 127.0.0.1:18888) | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:13000 | 13000 | /z/wangcy07/webarena/rootless-docker/data (volumes: webarena_verified_map_*) | WebArena | map (shared, read-only) | chuyue | N/A (Ruby+C++) | N on 3000/5000-5002/8080/8085 (127.0.0.1 only — use `ssh -L`); Y on :13000 (proxy → 127.0.0.1:13030) | N | Y | N |
| hilbit1.cis.upenn.edu | http://158.130.4.153:4399 | 4399 | /z/wangcy07/webarena/webarena-homepage | WebArena | homepage (shared) | chuyue | N/A (Python/Flask) | Y (proxy 0.0.0.0:4399 → 127.0.0.1:14399) | N | Y | N |

## Notes

- **All can also be accessed via localhost** on hilbit1 (the mutable sites bind `0.0.0.0`; wiki/map/homepage are reachable on localhost through the same public ports). Map's internal services (3000, 5000-5002, 8080, 8085) bind `127.0.0.1` only — forward them with `ssh -L` or the VS Code Ports panel if needed.
- **Port arithmetic**: `base + 100 × worker_id`. Bases: shopping 7770, shopping_admin 7780, reddit 9999, gitlab 8023. `shopping_admin` worker 3 is shifted **8080 → 8090** because 8080/8085 are reserved (`_HOST_PORT_RESERVED` in `mp/config.py`; on hilbit1 :8080 is also held by the unrelated `uber-benchmark-tileserver`).
- **Docker daemon**: rootless, socket `unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock`, data-root `/z/wangcy07/webarena/rootless-docker/data`. Source images came from tarballs (`/z/wangcy07/webarena/downloads/*.tar`) loaded into the daemon — there are no Dockerfiles built on hilbit1; the "absolute address" column points at the data-root / source artifacts accordingly.
- **Scanned? / Dynamic action captured?** = **N**: this is a fresh deployment; the security scan and dynamic-action capture (chuyue's downstream analysis steps) have not yet been run on hilbit1. **Dynamic code? = Y** is a property of the apps (identical images to hilbit2, where it is Y).

## Credentials (identical to hilbit2 — same golden images)

- **Forum / Reddit**: username `MarvelsGrantMan136`, password `test1234`
- **GitLab**: username `byteblaze`, password `hello1234` (default; if a reset/config changed it, depends on current container state)
- **Shopping (store user)**: username `emma.lopez@gmail.com`, password `Password.123`
- **Shopping Admin (Magento backend)** at `/admin`: username `admin`, password `admin1234`
- **Wikipedia, Map, Homepage**: no login required

## Quick health/timing reference (on-host, warm; load-dependent)

| Site | HTTP | Warm time |
|---|---|---|
| shopping | 200/302 | ~0.37 s |
| shopping_admin | 200/302 | ~0.41 s |
| reddit/forum | 200 | ~0.84 s |
| gitlab | 302 | ~0.06 s |
| wikipedia | 200 | ~0.008 s |
| map (home) | 200 | ~0.05 s |
| map tile (cached) | 200 | ~0.004 s |
| map nominatim | 200 | ~0.09 s |
| homepage | 200 | ~0.008 s |
