# GPU memory lifecycle

How this project treats VRAM as an owned resource, what it guarantees, and what
it explicitly does not.

The short version: every allocation has a named owner, every owner releases in
`finally`, and release is verified by measurement rather than assumed. Whether
CUDA actually returns the memory is the one thing that cannot be checked without
a GPU, and it is listed as pending.

---

## The four defects this replaced

Found by auditing the GPU-dependent paths rather than by hitting them, because
hitting them requires the hardware.

**1. `unload()` was not a release.** It set `self._model = None`. A HuggingFace
`pipeline` holds the model *and* the tokenizer; a returned tensor holds its
graph; an exception traceback holds every frame local, including the model. Any
one of those keeps weights resident after the owning attribute is cleared, and
the next load then fails with an OOM that appears to come from nowhere.

Now: move to CPU while the reference is still valid, drop every reference the
owner holds, collect, empty the cache, then **measure** what came back.

**2. The budget released reservations for memory that was still occupied.**
With `warm_start=True` the pipeline kept models loaded across documents, but the
`budget.phase()` context released the reservation on exit. Phase B then reserved
the summariser believing the full budget was free. On a 4 GB card at the declared
sizes: 550 + 2800 = 3350 MB actually resident against a 3195 MB budget that
reported 2800 MB used. A check that passes and an allocation that fails.

Now: `Pipeline._phase()` holds the reservation for as long as the model is
resident, and `close()` is what returns it.

**3. H1 would have held six model copies simultaneously.** The latency benchmark
cached one `Pipeline` per experimental condition, each with its own warm models.
Six conditions × ~2.8 GB ≈ 17 GB on a 4 GB card — the H1 experiment could not
have run at all.

Now: exactly one pipeline is resident; the previous is closed before the next
condition is constructed, and the memory is recorded at each transition.

**4. Timeouts were configured but never enforced.** `stage_timeout_s` and
`model_load_timeout_s` existed in `configs/*.yaml`, and `GPU_RUNBOOK.md` claimed
"nothing runs unbounded". No code read either value.

Now: `maor.execution.timeouts` enforces them, and a timeout is a distinct outcome
from a failure.

---

## What is guaranteed

| Guarantee | Mechanism | Verified by |
|---|---|---|
| A model is never loaded twice on one device | `ModelRegistry` keyed by `(device, checkpoint)` | `test_duplicate_load_of_same_checkpoint_is_refused` |
| Deliberate sharing is counted once | `allow_shared=True` returns the existing handle | `test_deliberate_sharing_returns_the_existing_handle` |
| A failed experiment releases its models | `model_scope` / `Pipeline.close()` in `finally` | `test_scope_releases_when_the_block_raises` |
| A failed experiment does not block the next | `ExperimentRunner` isolation | `test_failure_is_contained_and_the_next_experiment_still_runs` |
| Models leaked by a crash are reclaimed | `runner._verify_environment` releases stragglers | `test_models_left_resident_by_a_failed_run_are_released` |
| Nothing waits forever | `TimeoutGuard`, `run_with_timeout` | `test_slow_call_raises_timeout` |
| Retries are bounded | `RetryPolicy.max_attempts` | `test_retries_are_bounded` |
| An OOM is not retried identically | `retry_on_oom=False` by default | `test_oom_is_not_retried_by_default` |
| A workload that cannot fit fails fast | `plan_workload().raise_if_blocked()` | `test_blocked_plan_raises_on_demand` |
| Workers cannot oversubscribe the device | `recommend_worker_count` | `test_worker_count_is_capped_by_model_size` |
| An interrupted sequence resumes | `ExperimentRunner` checkpoint | `test_checkpoint_resume_skips_completed_experiments` |

---

## What is *not* guaranteed

Stated plainly, because a guarantee that does not hold is worse than none.

- **A wedged CUDA call cannot be interrupted from Python.** A kernel already
  dispatched to the device, or a C extension holding the GIL, will not respond to
  a watchdog. What is guaranteed is that the *process* reports which stage
  overran instead of waiting silently. Recovery from that state means restarting
  the process — the CUDA context cannot be reset in place.
- **Release verification has not been run against CUDA.** The measurement code is
  device-independent and tested, but whether `empty_cache()` returns what is
  expected on a T1000 is unverified until it runs there.
- **Estimated model sizes are estimates.** `models.*_estimated_vram_mb` in the
  config are declared expectations, not measurements. `vram-verify` reports the
  delta; correct the config from the measurement, never the reverse.
- **`reserved` staying high is normal.** The caching allocator keeps blocks for
  reuse. Release is judged on `allocated`; forcing `reserved` to zero would only
  make the next load slower.

---

## The execution lifecycle

`ExperimentRunner` runs each experiment through:

```
prepare environment  ->  release anything still registered, empty cache, reset peak
validate resources   ->  requires_gpu but no CUDA?  -> BLOCKED, not FAILED
execute              ->  under a deadline
save / checkpoint    ->  result written, progress recorded
release              ->  cleanup in finally, whatever the outcome
verify cleanup       ->  measure residual; flag if above tolerance
next
```

Outcomes are distinguished because they mean different things:

| Status | Meaning |
|---|---|
| `COMPLETED` | Ran and produced a result |
| `FAILED` | Raised; the next experiment still runs |
| `OOM` | Ran out of GPU memory; not retried with identical parameters |
| `TIMED_OUT` | Exceeded its deadline |
| `BLOCKED` | Requires hardware that is not present — implemented, not broken |
| `SKIPPED` | Already completed in a previous run (resumed from checkpoint) |

The sequence stops early only when continuing would be misleading: a critical
experiment failed, too many consecutive failures, or cleanup left more than
`residual_tolerance_mb` allocated. That last one matters — continuing after a
leak produces an OOM attributed to the wrong experiment.

---

## Research validity

Memory pressure must not quietly change what is measured.

**Adjusted automatically** — execution parameters only, because they change how
work is grouped, not what is computed:

- batch size (`plan_workload`, recorded in `execution_adjustments`)
- worker concurrency (`recommend_worker_count`)

**Never adjusted automatically** — these change the experiment, so a shortfall is
reported and the run stops:

- sample counts, sequence lengths, model choice, quantisation, chunk caps

Every adjustment is recorded in the result payload, so a reader can see that a
run used batch 4 rather than 8 and judge whether it matters.

Quantisation deserves a specific note. It is a memory-saving technique that
*does* change results, so it is configurable per model, defaults to `none` for
sentiment (leaving that measurement free of a quantisation confound), and its
cost is intended to be measured by the quantisation sweep rather than assumed.
It is set to `nf4` for the summariser in the T1000 config because a 3.8B model in
FP16 needs ~7.6 GB and will not fit — a case where the constraint forces the
choice, which is documented rather than silent.

---

## Instrumentation

Every experiment records:

```
gpu_name, total_vram_mb, allocated_before_mb, allocated_after_mb,
reserved_after_mb, peak_allocated_mb, peak_reserved_mb, residual_mb,
cleanup_clean, batch_size, concurrency, duration_s, status, failure_reason
```

`residual_mb` across a sequence is the leak detector: it should return to
roughly its starting value after each experiment. A value that climbs run over
run means something is retained, and the trace in `run_all_log.json` shows which
experiment started the climb.

---

## Commands

```bash
# What is on the device, what this process holds, and whether a workload fits
python -m maor.cli gpu-audit --config configs/gpu_t1000.yaml

# The whole sequence, with cleanup and verification between experiments
python -m maor.cli run-all --config configs/gpu_t1000.yaml

# Re-run only part of it (resumes from checkpoint by default)
python -m maor.cli run-all --only h2_summarisation --config configs/gpu_t1000.yaml

# Ignore the checkpoint and start over
python -m maor.cli run-all --no-resume --config configs/gpu_t1000.yaml
```

Global flags work before or after the subcommand.

---

## If something goes wrong

| Symptom | What it means | Action |
|---|---|---|
| `VRAMBudgetExceeded` before loading | The planner refused; nothing was allocated | Lower `vram.usable_fraction`, or release a resident model |
| `ModelAlreadyResident` / `DuplicateResidencyError` | Two components each tried to load the same checkpoint | Pass a handle to the existing instance; if sharing is intended, `allow_shared=True` |
| `cleanup DIRTY` in the summary | Memory was not returned after an experiment | Restart the process. Report which experiment first showed it |
| Status `OOM` | Genuinely ran out during execution | Reduce batch size or quantise further. It is deliberately not retried unchanged |
| Status `TIMED_OUT` | Exceeded the deadline | Raise `stage_timeout_s`, or investigate why that stage is slow |
| Status `BLOCKED` | No CUDA device | Expected off-GPU. Not a failure |
| `gpu-audit` shows other processes | Another process holds VRAM | Free it, or lower `usable_fraction` to fit alongside |
