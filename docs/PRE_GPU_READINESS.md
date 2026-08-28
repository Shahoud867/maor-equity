# Pre-GPU readiness gate

State of the workspace before GPU access. Every "verified" claim below names the
command or test that established it, and was actually run on this machine
(CPU-only: Intel Iris Xe, no CUDA).

---

## Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Codebase complete | Verified | 16 modules under `src/maor/`; `python -m maor.cli --help` lists 10 commands, all wired |
| Architecture implemented | Verified | One pipeline path serves both arms (`pipeline/orchestrator.py`); phase serialisation enforced by `VRAMBudget.phase()` |
| Data pipeline ready | Verified | `doctor` reports ECTSum available (495 records) and PhraseBank reachable; both loaders verify what they loaded |
| Experiments executable | Partly verified | H3 and chunk-filter **executed on real data**; H1/H2/VRAM implemented and unit-tested against injected executors, unrun (no CUDA) |
| Evaluation pipeline ready | Verified | 67 tests pass; ROUGE/BERTScore, bootstrap CIs, paired permutation tests all exercised |
| VRAM safeguards implemented | Verified | `TestVRAMBudget` (6 tests): over-budget refused pre-allocation, duplicate load refused, phase context releases, budget scales with the card |
| GPU configuration ready | Verified | `configs/gpu_t1000.yaml` loads and validates; `usable_fraction` 0.78 encodes measured overhead |
| Smoke tests ready | Verified | `maor.cli smoke` implemented with `configs/smoke.yaml`; CPU path exercised through sentiment, GPU path deferred |
| Documentation complete | Verified | Evidence policy, GPU runbook, audit response, paper notes, results status |
| Reproducibility setup complete | Verified | Provenance embedded in every result; integration check confirms 2/2 results load and 13/13 archived files refuse |
| Publication artifacts prepared | Partly verified | `paper/tables/h3_sentiment.tex` and `chunk_filter_curve.csv` generated from measured results; figures blocked by a broken local matplotlib |

---

## What was validated now, and how

**Validated by execution on real data**

- H3 over all 4,846 Financial PhraseBank sentences with real FinBERT checkpoints
  (495 s, `results/h3_sentiment.json`)
- ChunkFilter cost and coverage over 120 ECTSum documents
  (`results/chunk_filter_study.json`)
- ROUGE-L ≤ ROUGE-1 bound: 0 violations in 20,000 random pairs under the repo's
  own scorer
- The evidence policy itself: every file in `results/` loads with provenance;
  every one of 13 archived result files is refused

**Validated by test (67 passing)**

- Every audit finding has a regression test that fails on the old behaviour
- The full H1 analysis path — contrasts, Amdahl bound, hypothesis outcome —
  against an injected executor
- VRAM budget refusals, config cross-section validation, chunking invariants,
  statistical machinery

**Requires GPU — implemented, not run**

- H1 latency (2×2 factorial), H2 summarisation on ECTSum, VRAM verification,
  quantisation sweep. Listed as PENDING in `docs/RESULTS_STATUS.md`. No numbers
  are supplied for them.

**Requires a second machine**

- Cross-node communication measurement. `verify-cluster` is implemented;
  single-node results are valid for every other claim.

**Cannot be validated here**

- Figure rendering. The local `matplotlib` is an orphaned user-site install with
  no RECORD file, which pip cannot repair without deleting from the user's global
  environment. Tables are emitted as LaTeX and CSV instead. On the GPU node,
  `pip install matplotlib` in a clean environment resolves it.

---

## Exact sequence after connecting the GPU

Run in order. Steps 1–5 take about 45 minutes, mostly downloads.

```bash
# 1. Get the code
git clone https://github.com/Shahoud867/maor-equity.git
cd maor-equity

# 2. CUDA torch FIRST, then the rest — installing torch after can pull a CPU wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-gpu.txt

# 3. Confirm torch sees the GPU. Must print True.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Verify the environment. Must end 'READY for all experiments.'
python -m maor.cli doctor --config configs/gpu_t1000.yaml

# 5. Pre-download checkpoints (~2.5 GB)
python -m maor.cli fetch-models --config configs/gpu_t1000.yaml

# 6. Smoke test — do not skip. Must end 'SMOKE TEST PASSED'.
python -m maor.cli smoke --config configs/smoke.yaml

# 7. VRAM verification. Check declared_vs_measured; correct the config if it drifts.
python -m maor.cli vram-verify --config configs/gpu_t1000.yaml

# 8. H2, small first — inspect n_scaffolding_trimmed before committing hours
python -m maor.cli h2-summarisation --config configs/gpu_t1000.yaml --n-samples 5
python -m maor.cli h2-summarisation --config configs/gpu_t1000.yaml --n-samples 100

# 9. H1 with the full 2x2 ablation
python -m maor.cli h1-latency --config configs/gpu_t1000.yaml \
    --tickers AAPL MSFT GOOGL --repeats 5 --ablation full

# 10. Regenerate tables and the status page
python -m maor.cli report
```

Optional, after the above:

```bash
python -m maor.cli quantisation-sweep --config configs/gpu_t1000.yaml   # what 4-bit costs
ray start --head --port=6380 && python -m maor.cli verify-cluster       # distributed mode
```

Troubleshooting for each step is in [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md).

---

## What to watch for

Three results are worth reading carefully rather than recording:

1. **`vram-verify` → `declared_vs_measured`.** If measured residency differs from
   the declared figure by more than ~15%, update the config to the measured value.
   Correct the config, never the measurement.

2. **`vram-verify` → `coresidence_would_fit`.** This answers whether phase
   serialisation is actually necessary on the card, which is the paper's central
   design claim. If it says the models would fit co-resident, the claim needs
   restating.

3. **`h1-latency` → `parallelism_ceiling`.** If the measured speedup exceeds the
   Amdahl bound, the excess is not parallelism and must be attributed to its own
   factor. That substitution is what produced the original 1.72×.

Expect H1 to be unflattering. The previously measured runs gave 1.14× median with
the distributed arm slower on one of two documents, and the parallel fraction was
~4%. A modest or negative result there is the honest outcome and is more
defensible than the alternative.
