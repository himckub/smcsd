# SMC Experiment Scripts

Ad hoc entrypoints for SMC (Sequential Monte Carlo) speculative decoding
experiments on top of the standalone `smcsd/` implementation.

## Scripts

- **`accuracy_test_gsm8k.py`** — GSM8K accuracy benchmark (offline). Supports
  `smc_engine` (dedicated offline SMCEngine) and `baseline`
  (no speculative decoding).
- **`accuracy_test_gsm8k_http.py`** — GSM8K accuracy benchmark over an HTTP
  server (online serving). Launches the SMC HTTP server (`smcsd.http_server`)
  or connects to a running one (`--base-url`), then evals via concurrent native
  `/generate`. Same `#### <number>` scoring as the offline test, so results are
  directly comparable.
- **`quick_quality_check.py`** — Quick output quality sanity check
  (vanilla vs SMC) on a handful of hardcoded prompts.
- **`smc_profile_engine.py`** — Offline profiler harness for SMC. Use
  `--engine-kind smc_engine` to target the dedicated ``SMCEngine`` path;
  emits Chrome-compatible traces.
- **`tps_benchmark_scripts/`** — Throughput sweeps (shell scripts)
  across (gamma, n) pairs and batch sizes. See
  `tps_benchmark_scripts/BENCHMARK_CONFIGS.md` for details.

## Reproducing GSM8K Accuracy

```bash
source .venv/bin/activate

# Dedicated SMCEngine (recommended) — 8 particles, gamma=8 draft tokens
python scripts/accuracy_test_gsm8k.py --mode smc_engine -N 12 -g 8 --num-questions 400

# Baseline (no speculative decoding) for comparison
python scripts/accuracy_test_gsm8k.py --mode baseline --num-questions 400

# Custom models (Llama 3.1-8B target + Llama 3.2-1B draft)
python scripts/accuracy_test_gsm8k.py --mode smc_engine \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 8 --num-questions 200
```

Key flags for `accuracy_test_gsm8k.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `smc_engine` | `smc_engine` (dedicated SMCEngine) or `baseline` |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Target model |
| `--draft-model` | `meta-llama/Llama-3.2-1B-Instruct` | Draft model for SMC modes |
| `-N` / `--particles` | `4` | Number of SMC particles |
| `-g` / `--gamma` | `4` | Draft tokens per step |
| `--temperature` | `0.7` | Draft temperature |
| `--num-questions` | `80` | Number of GSM8K test questions |
| `--max-new-tokens` | `512` | Max generation length |
| `--batch-size` | `1` | Batch size for engine mode |
| `--mem-fraction-static` | `0.4` | GPU memory fraction (engine modes) |
| `--seed` | `None` | NumPy seed for reproducibility (question order only — GPU sampling is not seeded) |

## SMC Online Serving (HTTP)

SMC serves over HTTP via `smcsd/http_server.py`, which reuses sglang's standard
serving stack (TokenizerManager + DetokenizerManager + FastAPI) with the SMC
scheduler injected — no SMC source changes. Unlike the offline `SMCEngine`, the
HTTP server multiplexes concurrent requests.

```bash
source .venv/bin/activate
export FLASHINFER_WORKSPACE_BASE=/tmp/$USER-flashinfer   # shared-machine JIT cache

# Standard sglang endpoints become available (/generate, /v1/...)
python -m smcsd.http_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 8 --max-running-requests 16 --port 30000 --trust-remote-code
```

`--max-running-requests` is the number of concurrent SMC *groups*; it is expanded
by `(N+1)` internally (each group needs N+1 Req slots). Keep it modest relative to
`--mem-fraction-static` — CUDA-graph capture scales with the expanded value and can
OOM if set too high.

### GSM8K over HTTP

```bash
# Self-contained: launches the SMC server, evals, tears it down
python scripts/accuracy_test_gsm8k_http.py -N 8 -g 8 --num-questions 200 --parallel 16

# Against an already-running server (launched separately)
python scripts/accuracy_test_gsm8k_http.py --base-url http://127.0.0.1:30000

# Baseline (no spec) reference
python scripts/accuracy_test_gsm8k_http.py --mode baseline --num-questions 200
```

- Uses the zero-shot `#### <number>` format (no stop strings): SMC does not support
  stop strings, so sglang's few-shot `Question:/Answer:` harness can't drive it.
  Scoring matches `accuracy_test_gsm8k.py`, so HTTP and offline numbers are
  directly comparable (verified: offline 11/20 vs HTTP 9/20 on the same 20 GSM8K
  questions at temperature 0.7 — within sampling noise).
- SMC does not populate the EAGLE-style `avg_spec_accept_length` in `/server_info`,
  so accept length shows `n/a`; use output throughput as the speed metric.

## Throughput Sweeps

See `tps_benchmark_scripts/` for shell-based sweeps across batch sizes
and (gamma, n) configurations. Sweep scripts emit timestamped CSVs with
columns `method,gamma,n,tps,b`.

## Quick Quality Check

```bash
source .venv/bin/activate

python scripts/quick_quality_check.py --model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-path meta-llama/Llama-3.2-1B-Instruct --mode smc
```

## SMC Health Check

`scripts/health_check_smc.py` verifies that `quick_quality_check.py` is
runnable in SMC mode (it drives the current `SMCEngine` API). Two tiers,
exit code 0 = healthy / non-zero = broken:

```bash
source .venv/bin/activate

# Tier 1 — smoke (no GPU, seconds): asserts run_smc() wires SMCEngine, not the
# removed sgl.Engine(speculative_algorithm="SMC") path.
python scripts/health_check_smc.py

# Tier 2 — add a real GPU generation (loads 8B + 1B):
python scripts/health_check_smc.py --e2e
```

The `--e2e` preflight self-heals two common, non-code environment problems
(non-destructively; user-set values are respected): it points `CUDA_HOME` at a
valid CUDA toolkit if unset, and redirects the flashinfer JIT cache via
`FLASHINFER_WORKSPACE_BASE` when the shared `~/.cache/flashinfer` is stale
(e.g. a `build.ninja` pinned to a CUDA toolkit that no longer exists).

## Profiling

```bash
source .venv/bin/activate

python scripts/smc_profile_engine.py --engine-kind smc_engine \
    --output-dir /tmp/sglang-smc-profile
```

## Notes

- SMC runs through `SMCEngine` (dedicated offline path) or `smcsd.http_server`
  (online HTTP serving), both backed by `SMCScheduler` (subclass of the base
  `Scheduler`). There is no "engine-level" SMC via the regular `sgl.Engine`
  factory anymore.
- `FLASHINFER_WORKSPACE_BASE=/tmp/<user>-flashinfer` is often needed on
  shared machines when running GPU-backed experiments.
