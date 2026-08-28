# Audit response

Finding-by-finding record of what changed. Each row names the fix, where it
lives, and how it is verified. "Verified" means a test that fails if the defect
returns, or a measurement that has actually been taken — not an assertion that
the code looks right.

Status values:

- **Fixed & verified** — corrected, with a regression test that fails on the old behaviour
- **Fixed, pending GPU** — corrected and unit-tested; the measurement it enables needs CUDA
- **Superseded** — the artefact was removed from the evidence path and archived
- **Open** — not addressed, with the reason

---

## Critical

| # | Finding | Fix | Where | Verification | Status |
|---|---|---|---|---|---|
| C1 | Headline results generated, not measured, and presented as experimental | Two-class evidence taxonomy (`MEASURED` / `DERIVED`); provenance embedded in every result file; `read_result()` refuses files without it; no "estimated" class exists | `src/maor/provenance.py` | `TestC1Provenance` (5 tests) — a `DERIVED` result with no named inputs raises; a legacy hand-written result cannot be loaded | Fixed & verified |
| C1b | Generators and their outputs still in the evidence path | Moved to `archive/superseded_2026-08/` with a per-file account of what each produced and why it was removed | `archive/superseded_2026-08/README.md` | `results/` contains only provenance-stamped files | Superseded |
| C2 | The one real H1 measurement contradicted the published claim (1.14× measured vs 1.72× published; AAPL 1.69× *slower*) | Both original runs archived as historical evidence; H1 rebuilt as a factorial design that reports whatever it measures | `src/maor/evaluation/h1_latency.py` | `TestHypothesisReporting::test_h1_can_fail` — the analysis returns FAIL on a realistic workload | Fixed, pending GPU |
| C3 | ChunkFilter double-counted: B1 applied the identical filter, so it cannot explain a difference between arms | Filtering is now an independent factor in a 2×2 over {serial, distributed} × {filter on, off} | `src/maor/evaluation/h1_latency.py`, `src/maor/pipeline/orchestrator.py` | `TestContrasts::test_filter_effect_is_measured_on_both_arms` | Fixed, pending GPU |
| C3b | Baseline handicapped: B1 cold-loaded models while the distributed arm stayed warm | `warm_start` is an explicit factor; default comparison is warm-vs-warm; cold-start reported as its own contrast | `src/maor/evaluation/h1_latency.py` | `TestContrasts::test_warm_start_effect_is_its_own_factor` | Fixed, pending GPU |
| C4 | Hypothesis tests that could not fail (H2 compared a 0–1 delta against a tolerance of 1.0; H3 divergence guaranteed by construction) | `HypothesisTest` requires scale, units and achievable range, and `sanity_check()` flags a threshold outside that range; H2 restated as a non-inferiority test on the CI lower bound; H3 replaced with an accuracy test that can and does fail | `src/maor/evaluation/metrics.py`, `h2_summarisation.py`, `h3_sentiment.py` | `TestC4FalsifiableTests` (3 tests) — the exact original H2 configuration is flagged degenerate | Fixed & verified |
| C4b | Published ROUGE-L (0.32) exceeded ROUGE-1 (0.28), which no scorer produces | `RougeScores` raises on construction if `rouge_l > rouge_1` | `src/maor/evaluation/metrics.py` | `TestRougeInvariant` (2 tests); separately confirmed empirically — 0 violations in 20,000 random pairs under the repo's own scorer | Fixed & verified |
| C5 | T_comm measured GPU compute: `t_deserialize_ms` was 549,504 ms of Phi-3 generation; reported figure was 250 ms | Stage timings tagged by kind; only serialisation and transfer count as communication; transfer measured on an already-materialised object reference so no computation is in scope | `src/maor/pipeline/instrumentation.py` | `TestC5CommunicationAccounting` (3 tests) — a slow compute stage does not inflate the communication total | Fixed, pending GPU |
| C5b | gzip "80% bandwidth reduction" was never transmitted — payloads were decompressed before `ray.put()` | `measure_ray_communication(compress=True)` stores and transfers the compressed bytes; the claim is dropped unless compression is actually enabled | `src/maor/pipeline/instrumentation.py` | Code path stores `to_store = blob`; measured ratio reported alongside `compression_applied` | Fixed, pending GPU |

## Major

| # | Finding | Fix | Where | Verification | Status |
|---|---|---|---|---|---|
| M1 | "3-D sentiment" was two checkpoints; temporal reused the market pipeline object | Sharing is declared in `DimensionSpec.shares_checkpoint_with` and surfaced in every result; `n_distinct_checkpoints_loaded` is reported | `src/maor/agents/sentiment.py` | `TestM1DimensionHonesty` (2 tests) — asserts two distinct checkpoints, not three | Fixed & verified |
| M1b | The redundancy was never quantified | H3 now measures whether temporal labels are identical to market labels and reports which decision rules are consequently unsatisfiable | `src/maor/evaluation/h3_sentiment.py` | Measured: see `results/h3_sentiment.json` → `dimension_redundancy` | Fixed & verified |
| M2 | The regulatory row was computed from the placeholder string "No regulatory content detected." | Absent dimensions return `present=False` with no scores; the matrix marks them NaN; the guardrail states them as absent rather than neutral | `src/maor/agents/sentiment.py` | `TestM2NoPlaceholderContamination` (5 tests) | Fixed & verified |
| M3 | Nearest prior work (FinRobot, TradingAgents) uncited despite being vendored in the repo | Related-work positioning is a paper task, not a code task; recorded in `docs/PAPER_NOTES.md` with the differentiation the rewrite must make | `docs/PAPER_NOTES.md` | Not verifiable by test | Open (paper task) |
| M4 | Reported configuration did not match the code (FP16 vs NF4; VRAM trace predated the quantisation change by two days) | `vram-verify` records the resolved model configuration next to the measurement, and reports declared-vs-measured deltas | `src/maor/evaluation/vram_verify.py` | Result payload contains `resolved_model_configuration` and `declared_vs_measured` | Fixed, pending GPU |
| M4b | ChunkFilter cost stated as "~80 ms, essentially free"; the AAPL run logged 12,341 ms | `ChunkFilter.filter()` times itself and returns `elapsed_ms`; the study reports a cost distribution with CIs | `src/maor/data/chunking.py`, `evaluation/chunk_filter_eval.py` | Measured: 24.4 ms median at the default cap, 95% CI [22.98, 25.43] over 120 documents | Fixed & verified |
| M4c | "690 s saved" came from multiplying discarded chunks by an assumed 15 s each | The study emits chunk counts and states explicitly that the per-chunk cost must be measured on GPU before the arithmetic can be completed | `src/maor/evaluation/chunk_filter_eval.py` | `gpu_saving_arithmetic.status` reads `NOT COMPUTED` | Fixed & verified |
| M5 | No statistical validity: n=2, single run, no seeds, intervals or tests | Bootstrap CIs, paired bootstrap deltas, paired permutation tests, and an explicit sample-size interpretation attached to every result | `src/maor/evaluation/stats.py` | `TestM5Statistics` (6 tests) — n<3 is labelled "anecdote"; p-values are never exactly zero | Fixed & verified |
| M6 | Guardrail returned `UNRESOLVED / LOW` with 0.0 scores on parse failure, because `.get("confidence", 0.5)` never fired when the key existed with value 0.0 | `StanceParseResult.parsed` separates "model said zero" from "could not read the model"; unparsed stances yield `ASSESSMENT_FAILED` | `src/maor/agents/guardrail.py` | `TestM6GuardrailParseFailure` (5 tests) — including that 0.0 confidence is distinguishable from unparsed | Fixed & verified |
| M6b | Published summary contained leaked instruction scaffolding ("**Instruction 2 (More Difficult):**") | `clean_generation()` trims known scaffolding and the count is reported, so contamination is visible rather than published | `src/maor/agents/summarisation.py` | `TestSummaryContamination` (3 tests) — uses the verbatim shape of the original leak | Fixed & verified |
| M7 | 4-bit quantisation applied with no accuracy measurement | Quantisation is configurable per model and defaults to `none` for sentiment, so a measurement is of the model rather than of a quantisation artefact | `configs/*.yaml`, `src/maor/agents/*.py` | `TestConfigValidation::test_quantisation_values_are_constrained` | Fixed; sweep pending GPU |
| M8 | No error analysis; `qualitative_eval.py` and `scalability_eval.py` never run | H3 reports confusion matrices, low-confidence counts and divergence analysis; `sample_divergences()` extracts concrete cases | `src/maor/evaluation/h3_sentiment.py` | Present in `results/h3_sentiment.json` | Partially fixed |

## Minor

| # | Finding | Fix | Status |
|---|---|---|---|
| m1 | Reference [10] misattributed (BooookScore authors) and used to support a claim it does not make | Recorded in `docs/PAPER_NOTES.md` | Open (paper task) |
| m2 | Garbled prose in §III-B; abstract claims three forms of parallelism the limitations withdraw | Recorded in `docs/PAPER_NOTES.md` | Open (paper task) |
| m3 | `demo.html` shipped hardcoded outputs presented as pipeline results | `results/aapl_demo.json` archived; the demo must be regenerated from measured output or labelled a mock | Superseded |

---

## Newly discovered during remediation

Three problems that the audit could not have found without running the code.

**N1 — `yiyanghkust/finbert-tone` silently produced meaningless scores.**
Under `transformers` 5.x, `BertTokenizerFast(vocab_file=...)` builds a tokenizer
for this checkpoint that maps **every content word to `[UNK]`**, because the
vocabulary uses a non-standard special-token layout. The model then returns
`Neutral` at p≈1.00 for every input — plausible-looking numbers with no
information in them. A compatibility loader builds the tokenizer correctly, and
`verify_tokenizer()` now raises if more than 30% of ordinary financial English
resolves to `[UNK]`. Verified by `TestM2` fixtures and the H3 run.

**N2 — H3's dataset choice is train-on-test contamination.**
`ProsusAI/finbert` is fine-tuned on Financial PhraseBank, so evaluating it there
measures memorisation. The original design proposed exactly this. The threat is
attached to the result payload and printed by the CLI; the between-arm contrast
remains valid because both arms share the market model, but absolute accuracy
does not estimate field performance.

**N3 — the multi-dimensional design is structurally inert at sentence
granularity.** Because the temporal dimension is the market checkpoint applied to
the same text, deterministic inference forces identical labels, so any rule
requiring the two to disagree cannot fire. Measured: **985 of 985** routed
temporal labels identical to their market label (100.0%), and 0.00% divergence
over 4,846 sentences. This is the strongest finding produced so far and is a
genuine negative result.

**N3 follow-up — fixing the redundancy exposed a second, independent defect.**
The shared checkpoint was replaced with a genuinely different model
(`yiyanghkust/finbert-fls`, purpose-built for forward-looking-statement
detection) and re-measured on all 4,846 sentences. The fix worked mechanically:
three distinct checkpoints, temporal now returns a real, varied distribution
(82.0% specific, 12.3% not-FLS, 5.7% non-specific — not degenerate), and
divergence rose from the redundancy artefact of 0.00% to a real 1.548% (75
cases).

Accuracy, however, **decreased**: 88.96% → 88.22% (−0.74 pp, 95% CI
[−1.09, −0.41], paired permutation p = 0.0002). The CI excludes zero — this is a
statistically significant regression, not noise.

The cause is a second design defect, independent of the redundancy: the rule
("negative market + a *specific* forward commitment → soften to HOLD") assumes
specificity implies optimism. It does not — a company can commit to a specific
bad forecast, and finbert-fls's label says nothing about the forecast's
direction, only its concreteness. At 82% of routed sentences classified
specific, the rule fires broadly enough to soften many genuinely negative,
correctly-classified sentences into false HOLDs.

A second, more subtle issue compounds this: at sentence granularity, the
sentence routed as "temporal" *is* the sentence already scored for "market" —
they are literally the same text. The rule was designed for the deployed
pipeline's document-chunk granularity, where a temporal chunk is a genuinely
different span of a multi-chunk document than the chunks driving the aggregate
market score. Evaluated on single PhraseBank sentences (necessary because that
is where gold labels exist — see the granularity-mismatch validity threat), the
mechanism the rule was designed around does not operate the way it does in the
deployed pipeline. This is now the dominant open question for the temporal
dimension, and is not resolved by further threshold-tuning on this dataset,
which would be overfitting to a single test set rather than a fix.

**No further rule changes were made after seeing this result.** Continuing to
adjust the rule until the number looks better on the same held-out set is
exactly the failure mode this project's evidence policy exists to prevent. The
result is reported as measured, including the fact that it is a regression.

---

## Correction to the audit

The audit stated that the ChunkFilter cost was "off by ~154x", comparing the
paper's "~80 ms" against the 12,341 ms recorded in `results/aapl.json`. Both
numbers are real, but they measure different things, and the audit did not
distinguish them.

Measured directly over 120 ECTSum documents, the filter *algorithm* costs
**24.4 ms** (median; 95% CI [22.98, 25.43]) at the default cap of 12. The paper's
"~80 ms" was therefore the right order of magnitude for the algorithm. The
12,341 ms in the original log came from a timing window that enclosed more than
the filter — chunking with the Phi-3 tokenizer happens in the same block.

The finding that survives is narrower and still worth stating: the published
figure was asserted rather than measured, and the logged figure was mis-scoped.
Both are now measured, with the scope stated.

A second, more consequential thing the original never measured — what filtering
*costs in coverage* — is now quantified:

| cap | chunks kept | reduction | filter cost | vocabulary retained | document positions retained |
|---|---|---|---|---|---|
| 4 | 4.0 | 88.9% | 14.2 ms | 59.4% | 43.6% |
| 8 | 7.9 | 77.8% | 18.7 ms | 76.6% | 68.1% |
| **12 (default)** | **11.3** | **66.7%** | **24.4 ms** | **87.2%** | **83.0%** |
| 16 | 13.9 | 59.3% | 28.5 ms | 96.4% | 95.3% |
| 20 | 15.7 | 58.2% | 29.9 ms | 98.1% | 97.5% |
| 32 | 20.7 | 55.6% | 29.9 ms | 99.9% | 99.8% |

At the default cap the pipeline cannot reach **17% of each document**, and loses
13% of its vocabulary. Raising the cap to 16 recovers almost all of it for about
4 ms more CPU. That trade-off was never visible before, and it directly affects
the H2 quality comparison: the map-reduce arm is not summarising the whole
document either.

---

## GPU memory hardening (second pass)

A dedicated audit of the GPU-dependent paths, prompted by earlier runs stalling
on VRAM. Four defects, all found by reading rather than by hitting them — hitting
them requires the hardware, and three of the four would have made the GPU
experiments impossible rather than merely slow.

| # | Defect | Consequence | Fix | Verification |
|---|---|---|---|---|
| G1 | `unload()` only cleared the owning attribute | A HuggingFace `pipeline`, a returned tensor or an exception traceback keeps the model resident; the next load OOMs for no visible reason | Move to CPU while the reference is valid, drop every reference, collect, empty cache, then **measure** what returned | `test_release_flagged_dirty_when_memory_not_returned` |
| G2 | The budget released reservations for still-resident models | With `warm_start`, Phase A's reservation was returned while FinBERT stayed loaded, so Phase B reserved the summariser against space that did not exist — 3,350 MB actually resident against a 3,195 MB budget reporting 2,800 MB used | `Pipeline._phase()` holds the reservation while the model is resident; `close()` returns it | `TestVRAMBudget`, and `gpu-audit` now reports tracked vs measured residency |
| G3 | H1 cached one warm `Pipeline` per condition | Six conditions x ~2.8 GB ≈ 17 GB on a 4 GB card — **the H1 experiment could not have run at all** | Exactly one pipeline resident; the previous is closed before the next condition is built, with memory recorded at each transition | `test_h1_design.py`, plus `gpu_memory.marks` in the H1 payload |
| G4 | Timeouts were configured but never read | `GPU_RUNBOOK.md` claimed "nothing runs unbounded"; nothing enforced it | `maor.execution.timeouts` enforces `stage_timeout_s` and `model_load_timeout_s`; a timeout is a distinct outcome from a failure | `test_slow_call_raises_timeout`, `test_timeout_is_classified_and_cleanup_still_runs` |

Two further problems surfaced while testing the fixes:

**G5 — the watchdog could not raise its own exception.** `TimeoutGuard` raises
`TimeoutError_` into the main thread via `PyThreadState_SetAsyncExc`, which
instantiates the class with no arguments. `__init__` required `label` and
`seconds`, so every watchdog firing produced a `TypeError` and the timeout was
misreported as a generic failure. Both arguments now default.

**G6 — global CLI flags only worked before the subcommand.** Every example in
the GPU runbook is written as `<command> --config configs/gpu_t1000.yaml`, which
argparse rejected. Worse, the first attempted fix introduced a silent version of
the bug: a parent parser's defaults are re-applied by the subparser, so
`--config X <command>` was overwritten with `None` and fell back to the default
config — the T1000's 0.78 usable fraction silently became 0.85. Fixed with
`argparse.SUPPRESS`, and both orders are now regression-tested.

### Added infrastructure

- `src/maor/gpu/memory.py` — snapshots tracking **reserved** as well as
  allocated (reserved is what limits the next allocation), release verification,
  and a tracker whose `leaked_allocated_mb` makes accumulation across experiments
  visible as a number.
- `src/maor/gpu/lifecycle.py` — `ModelRegistry` keyed by `(device, checkpoint)`
  so duplicate residency is an error rather than an OOM; `model_scope` releasing
  in `finally`.
- `src/maor/gpu/limits.py` — pre-flight feasibility. Adjusts batch size and
  worker concurrency (execution parameters, which do not change what is
  computed) and refuses to touch sample counts, sequence lengths or quantisation
  (which do).
- `src/maor/execution/timeouts.py` — bounded waits, bounded retries, and OOM
  classification. Retrying an OOM with identical parameters is disabled by
  default because it fails identically and makes a run look hung.
- `src/maor/execution/runner.py` — the lifecycle: prepare, validate, execute
  under deadline, checkpoint, release, verify, next. A failed experiment releases
  its models and the next still runs; the sequence stops only when continuing
  would attribute an error to the wrong experiment.

### What this does not establish

The logic is tested; the CUDA behaviour is not. Whether `empty_cache()` actually
returns the expected memory on a T1000 is unverified until it runs there, and is
listed as pending. A wedged CUDA call still cannot be interrupted from Python —
the guarantee is that the process reports which stage overran rather than waiting
silently, and recovery from that state means restarting the process.

---

## What is not fixed

- **All GPU-dependent measurements.** H1, H2, VRAM verification and the
  quantisation sweep are implemented, unit-tested against injected executors, and
  documented in `docs/GPU_RUNBOOK.md` — but unrun, because the development
  machine has no CUDA device. They are listed as PENDING in
  `docs/RESULTS_STATUS.md` and no numbers are supplied for them.
- **Paper-level findings (M3, m1, m2).** Related-work positioning, citation
  correction and prose repair are writing tasks. They are recorded in
  `docs/PAPER_NOTES.md` rather than silently dropped.
- **Figure regeneration.** `matplotlib` is unusable in the development
  environment (an orphaned user-site install with no RECORD file, which pip
  cannot repair). Tables are generated as LaTeX and CSV instead; figures should
  be regenerated on the GPU node where the dependency installs cleanly.
