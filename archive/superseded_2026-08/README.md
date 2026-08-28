# Superseded material (archived 2026-08-28)

Nothing here is deleted, and nothing here may be cited.

These files were moved out of the evidence path during the remediation described
in `docs/EVIDENCE_POLICY.md`. They are kept because the audit findings depend on
them remaining inspectable: a reviewer asking "what exactly was wrong, and what
changed?" should be able to read the original code rather than take a summary on
trust.

`read_result()` raises `ProvenanceError` on every file in this directory, so none
of them can be loaded into a table or figure by accident.

---

## `quantitative/` — generators that produced the published numbers

These scripts computed the H1, H2 and H3 results reported in the paper. None of
them executed the system under test.

| File | What it produced | Why it was removed |
|---|---|---|
| `h1_amdahl_generator.py` | `h1_distributed_estimated.json` | Computed distributed latency as `B1_total / 1.723`, where `1.723 = amdahl(p=0.038) x 1.30 x 1.30`. Both 1.30 factors were hardcoded constants with comments describing them as measured. Because both tickers were the same division, they yielded identical speedups to three significant figures (1.7263, 1.7232) — the signature that gives the derivation away. |
| `h2_rouge_generator.py` | `h2_rouge_estimated.json` | ROUGE and BERTScore values were literals (`b2_rouge_1 = 0.28`); the map-reduce scores were the baseline plus a hand-chosen delta. No ECTSum document was scored. The published B2 row had ROUGE-L 0.32 above ROUGE-1 0.28, which is unreachable for F-measures over the same pairs. |
| `h3_sentiment_generator.py` | `h3_sentiment_estimated.json` | Sampled categorical labels from three hardcoded distributions with `random.seed(i)` and compared two hand-written `if` statements. No model, no text, no dataset. The 48% "divergence" was an arithmetic property of the two rules, and could not have been zero. |
| `confidence_validator.py` | `confidence_validation.json` | Reported 10/10 checks passed. The checks were tautologies over the generators' own output — e.g. `h1["result"] == "PASS"`, where the generator had written that string, and "speedup between 1.0 and 3.0" for a product of three constants each greater than 1. |
| `fig4_vram_generator.py` | `figures/fig4_vram_trace.png` | Rendered a VRAM trace from `vram_verify.json`, which predated the 4-bit quantisation change by two days. |

## `logs/` — result files

| File | Status |
|---|---|
| `h1_distributed_estimated.json` | Generated. Honestly named; the paper built on it was not. |
| `h1_latency_results.json` | Hand-authored. Its schema does not match `evaluation/latency_benchmark.py`, which is the script that supposedly produced it. |
| `h2_rouge_estimated.json`, `h2_rouge_results.json` | Generated / hand-authored. The `_results` file uses a 0–1 scale and different keys than `evaluation/rouge_eval.py` emits (0–100). |
| `h3_sentiment_estimated.json`, `h3_sentiment_results.json` | Generated / hand-authored. The `_results` file contains "accuracy vs ground truth 72.5% / 67.0%", which no script in the repository produces — the scenario generator has no ground truth at all. Those figures match the README's *expected* values. |
| `ablation_results.json` | Hand-authored. Contains the generator's constants (`warm_actor_factor: 1.3`) and does not match the schema of `evaluation/ablation_study.py`. The one ablation that would have tested the ChunkFilter claim (A1) was never run. |
| `confidence_validation.json` | Output of the tautological validator. |
| **`b1_results.json`** | **Genuinely measured**, and the only fully trustworthy quantitative artefact in the original project. Archived because it was measured on a since-changed configuration, not because it was fabricated. Worth re-running, not discarding. |
| `vram_verify.json` | Genuinely measured, but on 2026-04-21 01:14 — two days before FinBERT was switched to 4-bit NF4 on 2026-04-23. It describes a build that no longer exists. |

## `results/` — pipeline outputs

| File | Status |
|---|---|
| `aapl.json`, `msft.json` | **Genuinely measured** distributed runs (702.08 s and 777.12 s, 2026-04-21). These contradict the published 240.0 s and 738.0 s. They also contain the mis-instrumented `t_deserialize_ms: 549,504` — 549 seconds of GPU generation recorded as deserialisation — and a summary containing leaked instruction-tuning scaffolding. Archived as historical evidence; both should be re-measured on current code. |
| `aapl_demo.json` | Hand-authored fixture created for the demo dashboard, containing the generated 240.0 s total and a "bullish / HIGH" recommendation. The real runs return `UNRESOLVED / LOW`. |

---

## What replaced them

| Superseded | Replacement | Runnable now? |
|---|---|---|
| `h3_sentiment_generator.py` | `src/maor/evaluation/h3_sentiment.py` — real FinBERT over real Financial PhraseBank | Yes, CPU |
| `h2_rouge_generator.py` | `src/maor/evaluation/h2_summarisation.py` — real ROUGE/BERTScore over ECTSum | Needs GPU |
| `h1_amdahl_generator.py` | `src/maor/evaluation/h1_latency.py` — 2x2 factorial with measured contrasts | Needs GPU |
| `confidence_validator.py` | `tests/test_audit_regressions.py` — 55 tests that fail if a defect returns | Yes, CPU |
| `fig4_vram_generator.py` | `src/maor/evaluation/vram_verify.py` — measures the current build | Needs GPU |

See `docs/RESULTS_STATUS.md` for what has actually been measured to date.

---

## `legacy_packages/` — the original top-level Python packages

Superseded by `src/maor/`, which is a package rather than a set of top-level
directories on `sys.path`. Preserved because several of these files are the
*correct* implementations that were simply never run — the audit's point was not
that the harnesses were bad, but that their output was replaced by generated
numbers.

| Directory | Superseded by | Note |
|---|---|---|
| `agents/` | `src/maor/agents/` | `sentiment_agent.py` contains the `self._pipe_tmp = self._pipe_mkt` line (audit M1) and the placeholder-string fallback (M2). `guardrail_agent.py` contains the `.get("confidence", 0.5)` bug (M6). |
| `baselines/` | `src/maor/pipeline/orchestrator.py` | `b1_serial_pipeline.py:39` applies the same TF-IDF filter and 12-chunk cap as the distributed arm — the evidence for the double-counting finding (C3). |
| `evaluation/` | `src/maor/evaluation/` | **These harnesses were correct.** `rouge_eval.py`, `sentiment_eval.py`, `latency_benchmark.py` and `ablation_study.py` would have produced valid results if executed. Their output schemas are what proved the published `logs/*_results.json` files were not produced by them. |
| `optimization/` | `src/maor/data/chunking.py` | The TF-IDF algorithm is carried over unchanged; measurement and coverage reporting were added around it. |
| `communication/`, `profiling/` | `src/maor/pipeline/instrumentation.py` | Empty package stubs. |

## `figures/` — figures rendered from generated data

All eight figures were produced by `evaluation/generate_figures.py` reading
`logs/h1_latency_results.json`, `h2_rouge_results.json`, `h3_sentiment_results.json`
and `ablation_results.json` — every one of which is archived above as generated
or hand-authored. They must be regenerated from measured results.

## `misc/`

| File | Note |
|---|---|
| `demo.html` | Ships hardcoded sentiment vectors, summaries and BULLISH/BEARISH calls for AAPL/MSFT/TSLA presented as pipeline output. The real runs return `UNRESOLVED / LOW`. Must be regenerated from measured output or labelled a mock. |
| `cluster_config.yaml` | Superseded by `configs/*.yaml`, which are typed and validated. |
