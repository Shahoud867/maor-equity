# GPU runbook

Everything to run on the GPU machine, in order. Each step states what it does,
what success looks like, and what to do when it fails.

The design goal is that you never debug on the GPU node: every failure mode
below has a check that catches it early and an error message that names the fix.

**Time budget:** setup ~30 min (mostly model downloads), smoke test ~15 min,
full experiments 4–8 h depending on `n_repeats`.

---

## Step 0 — What you need

- An NVIDIA GPU with **≥ 4 GB VRAM** and a working driver
- Python 3.10–3.12
- ~15 GB free disk (model weights and the HuggingFace cache)
- Network access to huggingface.co

Everything is configurable, so a card larger than 4 GB works without edits —
`vram.total_mb: null` probes the device rather than assuming.

---

## Step 1 — Get the code

```bash
git clone https://github.com/Shahoud867/maor-equity.git
cd maor-equity
```

---

## Step 2 — Install dependencies

CUDA build of torch first — installing it after the other requirements can pull
a CPU-only wheel over it:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-gpu.txt
```

Confirm torch actually sees the GPU before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Expected:** a version string, `True`, and your GPU's name.

**If `False`:** you have a CPU-only torch build. `pip uninstall torch` and
reinstall from the cu121 index above. This is the single most common setup
failure and `doctor` reports it explicitly.

---

## Step 3 — Verify the environment

```bash
python -m maor.cli doctor --config configs/gpu_t1000.yaml
```

**Expected:** the GPU listed with its VRAM, a `[vram budget]` section showing
usable MB and whether the two models could co-reside, every core dependency
`ok`, ECTSum `available`, PhraseBank `reachable`, and a final line reading
`READY for all experiments.`

**If it says `BLOCKED`:** install the named core dependencies and re-run.

**If ECTSum is `missing`:** the file ships with the repository at
`data/ectsum/ectsum_test.jsonl` (495 records). Check it was not excluded by
`.gitignore` on your clone.

**If PhraseBank is `unreachable`:** network or rate limiting. Set `HF_TOKEN` to
a read token from huggingface.co/settings/tokens.

---

## Step 4 — Pre-download the models

Downloads ~2.5 GB. Doing it as its own step keeps the first real run from timing
out mid-download.

```bash
python -m maor.cli fetch-models --config configs/gpu_t1000.yaml
```

**Expected:** each checkpoint reported `cached`, with its on-disk size.

---

## Step 5 — Smoke test

Proves every code path end to end in minutes. **Do not skip this.**

```bash
python -m maor.cli smoke --config configs/smoke.yaml
```

**Expected:** each stage prints `ok`, a peak VRAM figure well under budget, and
a final `SMOKE TEST PASSED`. Results land in `results/smoke/` and are marked as
a wiring check, not evidence.

**If it fails on VRAM:** lower `vram.usable_fraction` (try 0.70) and re-run. The
error message names the model that did not fit and how much was free.

**If it hangs:** it should not — every stage has a timeout. If a stage times out
you get a stack trace naming the stage. Report the stage name.

---

## Step 6 — VRAM verification

The first real measurement, and the one that replaces the stale trace in the
paper (the published figure predated the 4-bit change by two days).

```bash
python -m maor.cli vram-verify --config configs/gpu_t1000.yaml
```

**Expected:** a stage-by-stage trace, `within_budget: true`, and a
`declared_vs_measured` block. Writes `results/vram_verification.json`.

**Look at `declared_vs_measured`.** If the measured residency differs from the
declared figure by more than ~15%, update
`models.*_estimated_vram_mb` in `configs/gpu_t1000.yaml` to the measured value
and re-run. Correct the config, never the measurement.

**Also look at `coresidence_would_fit`.** This answers the paper's design claim:
whether phase serialisation is actually necessary on this card.

---

## Step 7 — H2: summarisation quality

The longest run. ECTSum is already in the repository; nothing to download.

Start small to confirm generation quality before committing hours:

```bash
python -m maor.cli h2-summarisation --config configs/gpu_t1000.yaml --n-samples 5
```

Inspect `results/h2_summarisation.json` and check `n_scaffolding_trimmed`. If it
is high, the model is emitting instruction-tuning boilerplate into summaries —
the defect that reached the published AAPL summary. The cleaner removes it, but
a high count means prompts need work before the full run is worth doing.

Then the real run:

```bash
python -m maor.cli h2-summarisation --config configs/gpu_t1000.yaml --n-samples 100
```

**Expected:** ROUGE-1/2/L and BERTScore on the **0–100 scale** for both
map-reduce and the B2 single-pass baseline, with a paired bootstrap CI on the
ROUGE-L delta and a permutation p-value. Writes prediction files alongside the
scores so the numbers can be recomputed without re-running the model.

**Sanity check:** ROUGE-L must be ≤ ROUGE-1. `RougeScores` raises if it is not.

**Runtime:** roughly 20–40 min per 10 documents on a T1000. Budget accordingly,
or run under `nohup`/`tmux`.

---

## Step 8 — H1: latency and the parallelism ceiling

```bash
python -m maor.cli h1-latency --config configs/gpu_t1000.yaml --tickers AAPL MSFT GOOGL --repeats 5
```

**Expected:** per-stage timings tagged by kind (compute / communication / io /
model_load), a critical-path breakdown, the Amdahl bound at n=2,4,8, and
bootstrap CIs over repeats.

Two things to read carefully:

- **`communication_fraction`.** Communication now covers serialisation and
  transfer only. Blocking waits on remote computation are counted as compute.
  The previously published `t_comm_total_ms` of 562,906 was almost entirely
  Phi-3 generation.
- **`amdahl_bound`.** Reported as a bound, with no multipliers applied. If the
  measured speedup exceeds it, the excess comes from something that is not
  parallelism — warm residency, input reduction — and each needs its own control.

For the fair comparison the audit requires, run the 2×2:

```bash
python -m maor.cli h1-latency --config configs/gpu_t1000.yaml --ablation full
```

This crosses {serial, distributed} × {filter on, filter off}, which is the only
design that isolates the ChunkFilter's contribution. B1 already applied the same
filter, so any speedup attributed to filtering in the old analysis was
double-counted.

---

## Step 9 — Optional: quantisation sweep

Answers "what does 4-bit cost us?", which is the first question a reviewer asks
of a VRAM-constrained paper, and which was never measured.

```bash
python -m maor.cli quantisation-sweep --config configs/gpu_t1000.yaml
```

---

## Step 10 — Collect

```bash
python -m maor.cli report
```

Regenerates every table and figure from `results/`, and writes
`docs/RESULTS_STATUS.md` showing what is measured and what is still pending.
Anything without a provenance-stamped result file is listed as pending rather
than filled in.

---

## Distributed mode (optional)

Everything above runs single-node. Two-node Ray is only needed for the
cross-node communication measurement.

On the head node:

```bash
ray start --head --port=6380
python -m maor.cli verify-cluster --config configs/gpu_t1000.yaml
```

On the worker:

```bash
ray start --address='<HEAD_IP>:6380'
```

Then re-run H1 with `--set execution.mode=ray --set execution.ray_address=auto`.

**Note on scope.** Single-node results are valid for every claim except the
cross-node transfer cost. If the cluster is troublesome, run single-node and
report the communication measurement as not-yet-measured — that is a smaller
loss than the audit's original problem, which was reporting it anyway.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | CPU-only torch wheel | Reinstall from the cu121 index (Step 2) |
| `VRAMBudgetExceeded` on load | Model larger than the budget allows | Lower `vram.usable_fraction`, or `--set models.sentiment_quantisation=nf4` |
| `ModelAlreadyResident` | The shared model was loaded twice | A bug — the summariser handle should be passed, not re-created. Report it |
| CUDA OOM despite the budget check | Declared VRAM is too optimistic | Run `vram-verify`, then update the declared figures in the config |
| Run appears to hang | Should be impossible; every stage has a timeout | Wait for `stage_timeout_s`, then report which stage timed out |
| `TokenizerVerificationError` | A checkpoint's vocabulary is not loading | Expected for legacy repos; the compatibility path handles `finbert-tone`. Report the checkpoint name |
| `ProvenanceError: no provenance` | Reading a legacy hand-written result | Those files are not citable; regenerate with the CLI |
| Rate-limited by HuggingFace | Unauthenticated requests | `export HF_TOKEN=...` |

---

## What "done" looks like

After Steps 6–8, `results/` should contain provenance-stamped files for
`vram_verification`, `h2_summarisation` and `h1_latency`, each with a git commit,
hardware record and config SHA. `python -m maor.cli report` then produces the
tables, and `docs/RESULTS_STATUS.md` should show no remaining `PENDING` rows for
the claims the paper makes.
