# Paper notes

Open writing tasks carried over from the audit. These are not code problems and
cannot be closed by a test; they are recorded here so they are not silently
dropped.

---

## 1. Related work must be rewritten (audit M3)

The two closest systems are **vendored in this repository** and cited nowhere in
the submitted paper:

- **FinRobot: AI Agent for Equity Research and Valuation with Large Language
  Models** (arXiv 2411.08804) — a multi-agent LLM system for equity research with
  a lead agent orchestrating specialists. Same task, same architecture family.
  Present at `FinRobot/`.
- **TradingAgents: Multi-Agents LLM Financial Trading Framework**
  (arXiv 2412.20138) — specialist analyst agents plus **Bull and Bear researcher
  agents** and a risk-management team. This is essentially the Bull/Bear guardrail
  the paper claims as a contribution. Present at `TradingAgents/`.

Both must be cited and differentiated explicitly. The differentiation cannot be
"we built a multi-agent equity research system", because they did that first and
better. The defensible difference is the **hardware constraint**: neither
addresses running such a pipeline under a hard 4 GB VRAM budget, which is where
this project's real engineering contribution sits.

Also missing: the aspect-based financial sentiment literature (FinXABSA; ABSA on
FOMC minutes; structured multi-dimensional extraction from financial news).
"3-D sentiment" is a coarse, unvalidated instance of an established subfield and
cannot be presented as novel without engaging it.

## 2. Citation error (audit m1)

Reference [10] attributes BooookScore (arXiv 2310.00785) to "A. Chang and C. Xu".
The authors are Yapei Chang, Kyle Lo, Tanya Goyal and Mohit Iyyer.

It is also load-bearing for the claim that map-reduce summarisation is comparable
to single-pass. That paper studies book-length summarisation, where single-pass is
not an available option, so it does not support the claim. Either find a source
that does, or drop the claim and let H2 stand on its own measurement.

## 3. Prose defects (audit m2)

- §III-B contains a garbled passage ending "…Sep B (sequentially run on Node B)".
- The abstract claims three forms of parallelism (task, data, pipeline); the
  limitations section then states that data parallelism does not occur because
  the model serialises chunks on a single GPU. One of the two must go.

## 4. Claims that must be removed or restated

Carried from the audit, with the reason:

| Claim | Action | Why |
|---|---|---|
| "1.72× speedup / 42% latency reduction" | Remove | Never measured. Re-measure with `h1-latency` and report the outcome. |
| "80% bandwidth reduction" | Remove unless compression is enabled | Payloads were decompressed before transfer, so the reduction was never realised. |
| "Three forms of parallelism" | Restate | Contradicted by the paper's own limitations. |
| "3-D sentiment improves accuracy by 5.5 pp" | Remove | No experiment produced it. Measured value is 0.00 pp. |
| "48% direction divergence" | Remove | Measured value is 0.00%. |
| "ChunkFilter is the dominant source of speedup over B1" | Remove | B1 applied the same filter. |
| "> 4,000× return on computation" | Remove | Rests on an 80 ms cost; measured cost was 12,341 ms. |
| "Peak VRAM 3,261 MB" | Re-measure | The trace predates the 4-bit change by two days. |

## 4b. The temporal-dimension fix produced a second, more interesting negative result

After replacing the shared-checkpoint temporal dimension with a genuinely
independent model (`yiyanghkust/finbert-fls`), the redundancy is gone
(three distinct checkpoints, real 82/12/6% label distribution, 1.548% real
divergence) but accuracy **decreased** by 0.74 pp (CI [−1.09, −0.41] excludes
zero, p = 0.0002). This is now measured, not assumed — see
`docs/AUDIT_RESPONSE.md`, finding N3 follow-up.

This is worth a paragraph in the paper, not a footnote, because the mechanism is
identifiable and generalisable: the design conflated *specificity* of a forward
statement with its *optimism*. A model that detects "this is a concrete
forward-looking claim" says nothing about whether the claim is good or bad news,
and a rule that treats specificity as grounds to override a negative signal will
systematically mis-handle specific *bad* forecasts. Combined with a
sentence-level evaluation granularity mismatch (temporal-routed text and
market-scored text are literally the same sentence at this granularity, unlike
in the deployed document-chunk pipeline), this produced a small but
statistically significant regression.

**Do not re-tune the rule against this test set to chase a positive number.**
That would be the exact failure mode (P-hacking against a fixed evaluation set)
this project's evidence policy exists to prevent. If a corrected rule is
designed (e.g., combining specificity with the sentence's own polarity, which is
already computed as `market_label` at this granularity — no extra inference
needed), it should be pre-registered as a distinct, final hypothesis and
evaluated once, with the outcome reported regardless of which way it goes.

## 5. The thesis the evidence currently supports

The measured H3 result is a genuine negative finding, and it is more defensible
than anything the original claimed:

> On a corpus of real financial sentences, adding regulatory and temporal
> sentiment dimensions to a scalar baseline changed **zero** of 4,846 directional
> recommendations. The cause is structural: the temporal dimension is scored by
> the same checkpoint as the market dimension, so at sentence granularity the two
> labels are forced to agree, and the lexical router matched only 0.2% of
> sentences as regulatory.

Paired with the parallelism ceiling (once H1 is measured), this supports a
paper about **what does not work and why**, on severely constrained hardware —
which is honest, publishable in an applied venue, and does not require beating
FinRobot at its own task.

## 6. Threats to validity the paper must state

- `ProsusAI/finbert` is fine-tuned on Financial PhraseBank; absolute H3 accuracy
  measures memorisation, not generalisation. Only the between-arm contrast is
  interpretable.
- ECTSum is earnings-call transcripts; the deployed pipeline targets 8-K filings.
  The transfer gap is unmeasured.
- H3 runs at sentence granularity because that is where gold labels exist; the
  pipeline routes 512-token chunks. The granularity mismatch structurally
  suppresses the dimension interaction the design is meant to capture.
- Single hardware configuration; no evidence that timings generalise beyond a
  T1000 under WSL2.
- Neither H2 arm is a published ECTSum system, so the numbers cannot be positioned
  against the state of the art without reproducing one.
