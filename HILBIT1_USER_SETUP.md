# Hilbit1 — shared access for other users (no wangcy07 SSH key needed)

Any user with their **own** login on hilbit1.cis.upenn.edu can drive the WebArena
replicas + reset + orchestrator. They talk to the **local rootless-docker socket**
(mode 0666) — they do NOT need wangcy07s SSH key and never SSH as wangcy07.

## One-time: paste into your ~/.bashrc on hilbit1
```bash
export DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock
export PATH=/z/wangcy07/webarena-venv/bin:$PATH
export PYTHONPATH=/z/wangcy07/webarena-repo
```

## Sanity check (should list shopping/gitlab/forum containers)
```bash
docker ps --format "{{.Names}}" | head -5
```

## First-time per-user setup: your own result/auth dirs + config
```bash
mkdir -p $HOME/wa-results $HOME/wa-auth
cp /z/wangcy07/webarena-repo/mp/config.json $HOME/my-config.json
sed -i "s|\"result_dir\".*|\"result_dir\": \"$HOME/wa-results\",|" $HOME/my-config.json
sed -i "s|\"auth_root\".*|\"auth_root\": \"$HOME/wa-auth\",|"       $HOME/my-config.json
```

## Reset any replica (no SSH key needed — uses local socket)
```bash
python -m mp.reset_cli --site all --worker 0 --config $HOME/my-config.json
python -m mp.reset_cli --site shopping --worker 1 2 3      # specific workers
```

## Run a benchmark (point at your own GPU/LLM host)
```bash
export GPU_HOST=user@your-gpu-host          # your SSH key in ITS authorized_keys
cd /z/wangcy07/webarena-repo
bash mp/launch_tsa.sh                        # boots TSA + judge, opens tunnels
source mp/.inference_env
python -m mp.orchestrator --config $HOME/my-config.json --task_ids "22,24,47,48,126"
```

## What is shared vs private
- **Shared (read-only to you)**: the repo, venv, per-worker task configs, golden
  artifacts, and the docker socket. You cannot corrupt wangcy07s state.
- **Private (yours)**: result_dir + auth_root (point them at $HOME as above).
- **Containers are shared** — if two people reset/run the same worker at once they
  collide. Coordinate worker ids, or run on different workers.
- GitLab secrets in the golden tree are NOT world-readable; gitlab reset reads them
  inside the container (via the daemon), so it still works for you.
