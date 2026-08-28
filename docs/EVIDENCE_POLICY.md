# Evidence policy

This project previously published numbers that were generated rather than
measured, in files named like measurements, and cited them in a paper as
experimental results. This document defines the rules that now prevent that, and
the mechanisms that enforce them.

Read this before adding any number to the paper.

## The two evidence classes

Every file in `results/` carries a `_provenance` block with an `evidence_class`.
There are exactly two:

| Class | Meaning | May be cited as a result? |
|---|---|---|
| `MEASURED` | Produced by executing the system on real data | Yes |
| `DERIVED` | Computed analytically from named measured inputs | Only as an analysis of those inputs |

There is deliberately **no** class for estimated, projected, extrapolated or
simulated values. If a quantity has not been measured, it does not go in
`results/`. The paper reports it as not-yet-measured, with the command that
would produce it.

`DERIVED` results must name their inputs in `derived_from`. A derived number
with no inputs is an estimate, and `Provenance.validate()` raises rather than
writing it.

## What travels with every number

`write_result()` embeds provenance in the same file as the payload, so the two
cannot be separated by copying or re-committing:

- git commit, and whether the working tree was dirty
- UTC timestamp, hostname, username, exact command line
- Python version, platform, and versions of every library that can change a
  numeric result (torch, transformers, numpy, sklearn, scipy, rouge_score, …)
- the config SHA, so the run maps to an exact configuration
- the random seed
- the hardware the measurement was taken on
- caveats and validity threats that must not be dropped downstream

`read_result()` refuses to load a file without provenance. Legacy hand-written
result files therefore cannot be silently cited; they raise `ProvenanceError`.

## Rules

1. **No number in the paper without a file in `results/` that produced it.**
   Every table and figure must name the result file it was rendered from.

2. **No result file without a command that regenerates it.** If you cannot write
   the command, it is not reproducible and does not belong in the paper.

3. **A hypothesis test must be able to fail.** `HypothesisTest` requires the
   metric's scale, units and achievable range, and `sanity_check()` flags a
   threshold that lies outside that range. The original H2 test compared a delta
   on the 0–1 scale against a tolerance of 1.0, so every possible outcome passed;
   that specific defect is now a failing test in
   `tests/test_audit_regressions.py::TestC4FalsifiableTests`.

4. **Report the outcome, including when it refutes the hypothesis.** H3 currently
   FAILS on real data (0.00% divergence against a claimed 48%). That is the
   result. It is reported as the result.

5. **Validity threats travel with the number.** Where a run has a known threat —
   train-on-test contamination, an inert router, a granularity mismatch — the
   runner attaches it to the payload and the CLI prints it. Removing a threat
   from the paper requires removing it from the code that detects it.

6. **Measure, do not assume, the cost of an optimisation.** The ChunkFilter was
   described as "~80 ms, essentially free"; the measured cost was 12,341 ms.
   `ChunkFilter.filter()` now times itself and returns `elapsed_ms`.

7. **Do not convert measurements into savings with an assumed coefficient.**
   "690 s saved" came from multiplying discarded chunks by an assumed 15 s each.
   The chunk-filter study emits the chunk counts and states explicitly that the
   per-chunk cost must come from a GPU measurement before the arithmetic can be
   completed.

## What was removed, and where it went

The `quantitative/` directory contained generators that produced the published
H1, H2 and H3 numbers from hardcoded constants, and a "confidence validator"
that checked those constants against themselves. They are preserved for the
record under `archive/superseded_2026-08/` with a README explaining what each
produced, and are excluded from the evidence path.

Nothing was deleted. The audit findings depend on that code remaining
inspectable, and a reviewer asking "what exactly did you fix?" deserves to see it.

## Current evidence status

See `docs/RESULTS_STATUS.md` for the live table of what is measured, what is
pending GPU access, and the exact command for each.
