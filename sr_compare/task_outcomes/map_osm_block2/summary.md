# OSM block-2 — 58 map tasks on REAL OpenStreetMap (Qwen3-VL-32B)

Re-run of the WebArena `map` tasks (id ≥ 100, the 58 not covered by block1) pointing `map` at the
**real** `https://www.openstreetmap.org`. Same agent/scoring as block1.

| method | n_run | lenient_pass | official_pass | exec_errors |
|---|---|---|---|---|
| dense | 58 | 21 (36%) | 17 (29%) | 0 |
| tsa_tk128 | 58 | 6 (10%) | 4 (7%) | 0 |
| tsa_tk64 | 58 | 4 (7%) | 3 (5%) | 0 |
| tsa_tk32 | 58 | 6 (10%) | 4 (7%) | 0 |
| vortex_block | — | BLOCKED | BLOCKED | — |
| vortex_quest | — | BLOCKED | BLOCKED | — |

## Notes
- **Primary metric = lenient.** Real-OSM reference answers were annotated against the *self-hosted*
  WebArena OSM, so `official` (strict) is low **by construction**; n=58 is noise-prone.
- **dense > TSA here** is a real result: block2 (id≥100) is heavy on routing/distance tasks that need
  actual OSM interaction. Under sparse attention the agent more often fails to extract the route and
  returns "unable to complete"/empty (dense answered "4 min"/"3 min"/"3h28min" where TSA gave up).
  block1's id<100 subset was more knowledge-style, where TSA≈dense.
- TSA was built losslessly for H100/sm_90 (no CUDA upgrade): excluded the unused Blackwell TMA/XQA
  decode backends + the unused flashinfer prefill path; tree-sparse selection + flashinfer decode intact.

## vortex_block / vortex_quest — BLOCKED on this hardware
The vortex serve (vortex_torch sglang-0.5.9 fork, CUDA-12.8 build) **crashes under load** on this box:
`NCCL ... Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'` (SIGQUIT).
The agent therefore gets `Connection error` on every task → 0 valid LLM calls (an earlier auto-run
mis-scored these as lenient=9 on empty answers; those bogus outcomes were removed).
- Root cause: the cu128 / NCCL-2.29 stack issues CUDA calls the installed **GPU driver 560
  (CUDA 12.6)** cannot satisfy. Tried + ruled out: removing `--enable-deterministic-inference`;
  `NCCL_CUMEM_ENABLE=0 / NCCL_NVLS_ENABLE=0 / NCCL_P2P_DISABLE=1` (delayed but did not prevent the
  crash); single-instance serving. The serve answers a trivial curl request but dies under the
  agent's sustained real workload.
- To run vortex here: upgrade the GPU driver to one matching CUDA 12.8 (≥ ~570), or run on a box whose
  driver matches the vortex cu128 build. (Driver upgrade intentionally not done — needs root + reboot
  on this shared machine.)
- dense + TSA are unaffected (they don't use the vortex stack).
