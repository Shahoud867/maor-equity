"""Regression tests for each finding in the MAOR-Equity audit.

Every test here corresponds to a specific defect that reached the submitted
paper. They are written so that reintroducing the defect fails the suite, which
is the only durable way to keep a fix from being undone later.

Test names carry the audit finding id (C1-C5, M1-M7).
"""

from __future__ import annotations

import math

import pytest

from maor import hardware
from maor.agents.guardrail import (
    GuardrailAgent,
    Recommendation,
    StanceParseResult,
    parse_stance,
)
from maor.agents.sentiment import DimensionRouter, SentimentBundle, build_matrix
from maor.agents.summarisation import clean_generation
from maor.config import Config, ConfigError
from maor.data.chunking import ChunkFilter, Chunk, chunk_document, coverage_stats
from maor.evaluation import stats
from maor.evaluation.metrics import HypothesisTest, RougeScores
from maor.provenance import (
    EvidenceClass,
    Provenance,
    ProvenanceError,
    read_result,
    write_result,
)


# ---------------------------------------------------------------------------
# C1 — results must carry provenance and an explicit evidence class
# ---------------------------------------------------------------------------


class TestC1Provenance:
    def test_derived_result_must_name_its_inputs(self):
        """A derived number with no named inputs is an estimate. Reject it."""
        prov = Provenance(
            evidence_class=EvidenceClass.DERIVED,
            experiment="amdahl_bound",
            hardware={"has_cuda": False},
        )
        with pytest.raises(ProvenanceError, match="derived_from"):
            prov.validate()

    def test_measured_result_must_record_hardware(self):
        prov = Provenance(evidence_class=EvidenceClass.MEASURED, experiment="h1")
        with pytest.raises(ProvenanceError, match="hardware"):
            prov.validate()

    def test_there_is_no_estimated_evidence_class(self):
        """The taxonomy deliberately offers no way to label a guess as a result."""
        assert {e.value for e in EvidenceClass} == {"MEASURED", "DERIVED"}

    def test_reading_a_result_without_provenance_is_an_error(self, tmp_path):
        """Hand-written result files must not be silently citable."""
        legacy = tmp_path / "h1_latency_results.json"
        legacy.write_text('{"speedup": 1.72, "H1_passed": true}', encoding="utf-8")
        with pytest.raises(ProvenanceError, match="no provenance"):
            read_result(legacy)

    def test_roundtrip_preserves_evidence_class_and_payload(self, tmp_path):
        prov = Provenance(
            evidence_class=EvidenceClass.MEASURED,
            experiment="demo",
            hardware={"has_cuda": False, "cpu_count": 4},
        )
        out = write_result(tmp_path / "r.json", {"value": 42}, provenance=prov)
        payload, loaded = read_result(out)
        assert payload["value"] == 42
        assert loaded.evidence_class is EvidenceClass.MEASURED
        assert loaded.experiment == "demo"


# ---------------------------------------------------------------------------
# C3 — the baseline must be configurable so the filter is not double-counted
# ---------------------------------------------------------------------------


class TestC3FilterAttribution:
    def test_filter_is_an_independent_factor(self):
        """ChunkFilter must be applicable to either arm, or it cannot be ablated.

        B1 applied the identical filter and cap, so attributing speedup over B1
        to the filter double-counts it. The 2x2 design requires the filter to be
        switchable independently of serial/distributed execution.
        """
        chunks = [
            Chunk(chunk_id=i, text=f"segment {i} " + "revenue growth margin " * 20,
                  token_count=60, start_token=i * 10)
            for i in range(30)
        ]
        filt = ChunkFilter(max_chunks_8k=12)

        filtered = filt.filter(chunks, filing_type="8-K")
        assert filtered.n_after <= 12
        assert filtered.n_before == 30

        unfiltered = filt.filter(chunks, filing_type="8-K", max_chunks=len(chunks))
        assert unfiltered.n_after == 30
        assert unfiltered.reduction_pct == 0.0

    def test_filter_reports_measured_cost_not_assumed(self):
        chunks = [
            Chunk(chunk_id=i, text=f"chunk {i} " + "alpha beta gamma " * 30,
                  token_count=90, start_token=i * 10)
            for i in range(40)
        ]
        result = ChunkFilter(max_chunks_8k=12).filter(chunks)
        assert result.elapsed_ms > 0.0, "filter must time itself, not assume ~80ms"
        assert "tfidf" in result.method or "uniform" in result.method

    def test_filter_reports_what_it_discarded(self):
        """Coverage statistics make the quality cost of filtering measurable."""
        chunks = [
            Chunk(chunk_id=i, text=f"unique{i} " + "shared words here " * 20,
                  token_count=63, start_token=i * 10)
            for i in range(30)
        ]
        result = ChunkFilter(max_chunks_8k=8).filter(chunks)
        cov = result.coverage
        assert cov["vocabulary_retained_pct"] is not None
        assert cov["document_positions_retained_pct"] is not None
        assert 0 <= cov["vocabulary_retained_pct"] <= 100


# ---------------------------------------------------------------------------
# C4 — hypothesis tests must be able to fail
# ---------------------------------------------------------------------------


class TestC4FalsifiableTests:
    def test_the_original_h2_tolerance_is_flagged_as_degenerate(self):
        """|ROUGE-L delta| < 1.0 on the 0-1 scale passes for every possible result."""
        broken = HypothesisTest(
            hypothesis_id="H2",
            claim="map-reduce ROUGE-L is non-inferior to single-pass",
            metric_name="abs_rouge_l_delta",
            scale="0-1",
            units="ROUGE fraction",
            observed=0.01,
            threshold=1.0,
            comparison="<",
            falsifiable_range=(0.0, 1.0),
        )
        warnings = broken.sanity_check()
        assert warnings, "an unfalsifiable test must be flagged"
        message = " ".join(warnings).lower()
        assert "every possible outcome passes" in message or "cannot fail" in message

    def test_correctly_scaled_h2_tolerance_is_not_flagged(self):
        ok = HypothesisTest(
            hypothesis_id="H2",
            claim="map-reduce ROUGE-L is non-inferior to single-pass",
            metric_name="rouge_l_delta",
            scale="0-100",
            units="ROUGE points",
            observed=-0.4,
            threshold=-1.0,
            comparison=">=",
            falsifiable_range=(-100.0, 100.0),
        )
        assert ok.sanity_check() == []
        assert ok.passed is True

    def test_a_test_can_actually_fail(self):
        t = HypothesisTest(
            hypothesis_id="H3b",
            claim="multidimensional beats scalar",
            metric_name="accuracy_delta",
            scale="percentage points",
            units="pp",
            observed=0.0,
            threshold=0.0,
            comparison=">",
            falsifiable_range=(-100.0, 100.0),
        )
        assert t.passed is False


# ---------------------------------------------------------------------------
# C4b — ROUGE-L cannot exceed ROUGE-1
# ---------------------------------------------------------------------------


class TestRougeInvariant:
    def test_rouge_l_above_rouge_1_is_rejected(self):
        """The published B2 row had R-L 0.32 > R-1 0.28, which no scorer produces."""
        with pytest.raises(ValueError, match="impossible"):
            RougeScores(rouge_1=28.0, rouge_2=12.0, rouge_l=32.0, n=100)

    def test_valid_ordering_is_accepted(self):
        s = RougeScores(rouge_1=28.0, rouge_2=12.0, rouge_l=24.0, n=100)
        assert s.rouge_l <= s.rouge_1


# ---------------------------------------------------------------------------
# C5 — communication accounting must exclude compute
# ---------------------------------------------------------------------------


class TestC5CommunicationAccounting:
    def test_stage_kinds_are_disjoint_and_validated(self):
        from maor.pipeline.instrumentation import StageRecorder

        rec = StageRecorder()
        with rec.stage("summarise", kind="compute"):
            pass
        with rec.stage("put_chunks", kind="communication"):
            pass
        totals = rec.by_kind()
        assert set(totals) == {"compute", "communication"}

        with pytest.raises(ValueError, match="unknown stage kind"):
            with rec.stage("bogus", kind="transfer"):
                pass

    def test_blocking_compute_is_not_counted_as_communication(self):
        """A slow compute stage must not inflate the communication total."""
        import time as _time

        from maor.pipeline.instrumentation import StageRecorder

        rec = StageRecorder()
        with rec.stage("phi3_map_reduce", kind="compute"):
            _time.sleep(0.05)
        with rec.stage("put_payload", kind="communication"):
            pass

        totals = rec.by_kind()
        assert totals["compute"] >= 0.05
        assert totals["communication"] < 0.01
        summary = rec.to_dict()
        assert summary["communication_fraction"] < 0.5

    def test_amdahl_bound_is_reported_without_multipliers(self):
        from maor.pipeline.instrumentation import StageRecorder, critical_path_analysis

        rec = StageRecorder()
        with rec.stage("sentiment", kind="compute", parallelisable=True):
            pass
        with rec.stage("summarise", kind="compute", parallelisable=False):
            pass

        analysis = critical_path_analysis(rec)
        assert "amdahl_bound" in analysis
        for value in analysis["amdahl_bound"].values():
            assert value >= 1.0
        # The bound must be a bound: no warm-start or filtering multiplier here.
        assert "warm_actor_factor" not in analysis
        assert "data_parallelism_factor" not in analysis


# ---------------------------------------------------------------------------
# M1 — the temporal dimension shares the market checkpoint, and says so
# ---------------------------------------------------------------------------


class TestM1DimensionHonesty:
    def test_temporal_declares_it_shares_the_market_checkpoint(self):
        bundle = SentimentBundle(device="cpu")
        specs = {s.name: s for s in bundle.specs}
        assert specs["temporal"].shares_checkpoint_with == "market"
        assert specs["temporal"].checkpoint == specs["market"].checkpoint
        assert specs["regulatory"].shares_checkpoint_with is None

    def test_there_are_two_distinct_checkpoints_not_three(self):
        bundle = SentimentBundle(device="cpu")
        distinct = {s.checkpoint for s in bundle.specs}
        assert len(distinct) == 2, "the '3-D' design uses two sets of weights"


# ---------------------------------------------------------------------------
# M2 — an absent dimension is absent, never a placeholder sentence
# ---------------------------------------------------------------------------


class TestM2NoPlaceholderContamination:
    def test_router_emits_no_placeholder_text(self):
        """The original emitted 'No regulatory content detected.' and scored it."""
        routed = DimensionRouter().route(
            [{"text": "Revenue increased twelve percent this quarter."}]
        )["routed"]
        assert routed["regulatory"] == []
        for bucket in routed.values():
            for text in bucket:
                assert "No regulatory content detected" not in text
                assert "No forward-looking statements detected" not in text

    def test_absent_dimension_is_marked_absent_not_neutral(self):
        bundle = SentimentBundle(device="cpu")
        result = bundle.classify([], "regulatory")
        assert result.present is False
        assert result.mean_vector is None
        assert result.n_texts == 0

    def test_matrix_marks_missing_rows_as_nan(self):
        bundle = SentimentBundle(device="cpu")
        results = {
            "market": bundle.classify([], "market"),
            "regulatory": bundle.classify([], "regulatory"),
            "temporal": bundle.classify([], "temporal"),
        }
        matrix = build_matrix(results)
        assert matrix.n_present == 0
        assert all(math.isnan(v) for row in matrix.rows for v in row)
        assert matrix.direction("regulatory") is None

    def test_router_detects_real_regulatory_content(self):
        routed = DimensionRouter().route(
            [{"text": "The SEC opened an enforcement investigation into the filing."}]
        )["routed"]
        assert len(routed["regulatory"]) == 1

    def test_router_reports_coverage(self):
        out = DimensionRouter().route(
            [
                {"text": "The SEC issued a penalty."},
                {"text": "Revenue rose."},
                {"text": "We expect growth next quarter."},
            ]
        )
        cov = out["coverage"]
        assert cov["n_chunks"] == 3
        assert cov["regulatory_pct"] == pytest.approx(33.33, abs=0.1)
        assert cov["regulatory_absent"] is False


# ---------------------------------------------------------------------------
# M6 — a parse failure must not masquerade as a verdict
# ---------------------------------------------------------------------------


class TestM6GuardrailParseFailure:
    def test_unparseable_stance_is_marked_unparsed(self):
        result = parse_stance("I cannot produce JSON for this request.")
        assert result.parsed is False
        assert result.confidence is None
        assert result.parse_error

    def test_parse_failure_yields_assessment_failed_not_unresolved(self):
        """The original returned UNRESOLVED/LOW with 0.0/0.0 scores on parse failure."""
        agent = GuardrailAgent(model=None)
        verdict = agent.arbitrate(
            bull=StanceParseResult(parsed=False, parse_error="no JSON object found"),
            bear=StanceParseResult(parsed=False, parse_error="no JSON object found"),
            technical={"rsi": 50.0},
        )
        assert verdict["recommendation"] == Recommendation.ASSESSMENT_FAILED
        assert verdict["recommendation"] != Recommendation.UNRESOLVED
        assert verdict["bull_score"] is None
        assert verdict["bear_score"] is None
        assert "bull" in verdict["unparsed_stances"]

    def test_genuine_balance_is_unresolved_with_real_scores(self):
        agent = GuardrailAgent(model=None)
        verdict = agent.arbitrate(
            bull=StanceParseResult(parsed=True, confidence=0.50, signals=["a"]),
            bear=StanceParseResult(parsed=True, confidence=0.52, signals=["b"]),
            technical={"rsi": 50.0},
        )
        assert verdict["recommendation"] == Recommendation.UNRESOLVED
        assert verdict["bull_score"] is not None

    def test_decisive_margin_produces_a_recommendation(self):
        agent = GuardrailAgent(model=None)
        verdict = agent.arbitrate(
            bull=StanceParseResult(parsed=True, confidence=0.90, signals=["strong"]),
            bear=StanceParseResult(parsed=True, confidence=0.20, signals=["weak"]),
            technical={"rsi": 50.0},
        )
        assert verdict["recommendation"] == Recommendation.BULLISH
        assert verdict["confidence"] == "HIGH"

    def test_zero_confidence_is_distinguishable_from_unparsed(self):
        """The exact bug: .get('confidence', 0.5) never fired because 0.0 existed."""
        parsed_zero = parse_stance('{"direction":"bull","confidence":0.0,"signals":[]}')
        assert parsed_zero.parsed is True
        assert parsed_zero.confidence == 0.0

        unparsed = parse_stance("garbage")
        assert unparsed.parsed is False
        assert unparsed.confidence is None


# ---------------------------------------------------------------------------
# M6b — generated summaries must not carry instruction scaffolding
# ---------------------------------------------------------------------------


class TestSummaryContamination:
    def test_instruction_scaffolding_is_trimmed(self):
        """Verbatim shape of the leak in results/aapl.json."""
        raw = (
            "Executive Summary:\nApple filed an 8-K.\n\n"
            "**Instruction 2 (More Difficult):**\n\nAnalyze the provided SEC filing"
        )
        cleaned, modified = clean_generation(raw)
        assert modified is True
        assert "Instruction 2" not in cleaned
        assert "Apple filed an 8-K." in cleaned

    def test_clean_output_is_untouched(self):
        raw = "Executive Summary: revenue grew 12% with stable margins."
        cleaned, modified = clean_generation(raw)
        assert modified is False
        assert cleaned == raw

    def test_chat_template_tokens_are_trimmed(self):
        cleaned, modified = clean_generation("The answer.<|end|> extra junk")
        assert modified is True
        assert cleaned == "The answer."


# ---------------------------------------------------------------------------
# VRAM — the hard constraint
# ---------------------------------------------------------------------------


class TestVRAMBudget:
    def test_over_budget_request_is_refused_before_allocation(self):
        budget = hardware.VRAMBudget(total_mb=4096.0, usable_fraction=0.85)
        with pytest.raises(hardware.VRAMBudgetExceeded):
            budget.reserve("huge_model", 5000.0)

    def test_duplicate_model_load_is_refused(self):
        """Loading the shared summariser twice is what blows a 4 GB budget."""
        budget = hardware.VRAMBudget(total_mb=8192.0)
        budget.reserve("phi3", 2800.0)
        with pytest.raises(hardware.ModelAlreadyResident, match="already resident"):
            budget.reserve("phi3", 2800.0)

    def test_phase_context_releases_on_exit(self):
        budget = hardware.VRAMBudget(total_mb=4096.0)
        with budget.phase("finbert", 550.0):
            assert budget.reserved_mb == 550.0
        assert budget.reserved_mb == 0.0

    def test_phase_serialisation_enables_what_coresidence_forbids(self):
        budget = hardware.VRAMBudget(total_mb=4096.0, usable_fraction=0.80)
        assert budget.usable_mb == pytest.approx(3276.8)
        with budget.phase("finbert", 1500.0):
            with pytest.raises(hardware.VRAMBudgetExceeded):
                budget.reserve("phi3", 2800.0)
        with budget.phase("phi3", 2800.0):
            assert budget.reserved_mb == 2800.0

    def test_budget_scales_with_the_card_rather_than_hardcoding_4gb(self):
        small = hardware.VRAMBudget(total_mb=4096.0, usable_fraction=0.85)
        large = hardware.VRAMBudget(total_mb=16384.0, usable_fraction=0.85)
        assert large.usable_mb == pytest.approx(4 * small.usable_mb)

    def test_release_frees_the_allowance(self):
        budget = hardware.VRAMBudget(total_mb=4096.0)
        budget.reserve("a", 1000.0)
        budget.release("a")
        assert budget.available_mb == budget.usable_mb


# ---------------------------------------------------------------------------
# Config — cross-section validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_config_is_valid(self):
        Config.load().validate()

    def test_coresidence_without_phase_serialisation_is_rejected(self):
        """Precisely the condition that caused the original stalls.

        Note the margin is thin: at the declared 550 + 2800 MB and
        usable_fraction=0.85, the two models *nominally* fit on a 4 GB card
        (3,350 vs 3,482 MB usable). They did not fit in practice, because CUDA
        context and KV cache growth are invisible to those declared figures.
        That is why phase serialisation is the default and why
        ``usable_fraction`` is tuned down in configs/gpu_t1000.yaml rather than
        relying on the nominal arithmetic.
        """
        cfg = Config.load()
        cfg.vram.total_mb = 4096.0
        cfg.vram.phase_serialised = False
        cfg.models.sentiment_estimated_vram_mb = 700.0
        cfg.models.summarizer_estimated_vram_mb = 2900.0  # 3,600 > 3,482 usable
        with pytest.raises(ConfigError, match="phase serialisation"):
            cfg.validate()

    def test_nominal_fit_does_not_imply_practical_fit(self):
        """The declared figures fit; the measured reality on a T1000 did not."""
        cfg = Config.load()
        cfg.vram.total_mb = 4096.0
        cfg.vram.phase_serialised = False
        cfg.models.sentiment_estimated_vram_mb = 550.0
        cfg.models.summarizer_estimated_vram_mb = 2800.0
        cfg.validate()  # passes on nominal arithmetic

        # Tightening the usable fraction to reflect measured overhead flips it.
        cfg.vram.usable_fraction = 0.80
        with pytest.raises(ConfigError, match="phase serialisation"):
            cfg.validate()

    def test_model_larger_than_the_whole_budget_is_rejected(self):
        cfg = Config.load()
        cfg.vram.total_mb = 2048.0
        cfg.models.summarizer_estimated_vram_mb = 9000.0
        with pytest.raises(ConfigError, match="cannot help"):
            cfg.validate()

    def test_unknown_config_key_is_an_error(self):
        cfg = Config.load()
        with pytest.raises(ConfigError, match="unknown config key"):
            cfg.set("vram.nonexistent_option", 1)

    def test_stride_must_be_smaller_than_window(self):
        cfg = Config.load()
        cfg.chunking.stride_tokens = 512
        cfg.chunking.window_tokens = 512
        with pytest.raises(ConfigError, match="stride_tokens"):
            cfg.validate()

    def test_quantisation_values_are_constrained(self):
        cfg = Config.load()
        cfg.models.summarizer_quantisation = "fp4-ish"
        with pytest.raises(ConfigError, match="quantisation"):
            cfg.validate()


# ---------------------------------------------------------------------------
# M5 — statistical validity
# ---------------------------------------------------------------------------


class TestM5Statistics:
    def test_bootstrap_interval_brackets_the_point_estimate(self):
        values = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5]
        ci = stats.bootstrap_ci(values, n_resamples=2000, seed=1)
        assert ci.lower <= ci.point <= ci.upper
        assert ci.n == len(values)

    def test_tiny_samples_are_labelled_as_anecdote(self):
        """n=2 was the original H1 sample size."""
        assert stats.interpret_sample_size(2)["evidence_level"] == "anecdote"
        assert stats.interpret_sample_size(50)["evidence_level"] == "adequate"

    def test_paired_delta_ci_spans_zero_for_identical_systems(self):
        a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        ci = stats.paired_bootstrap_delta(a, list(a), n_resamples=2000, seed=3)
        assert ci.point == 0.0
        assert ci.lower <= 0.0 <= ci.upper

    def test_permutation_p_value_is_never_exactly_zero(self):
        a = [5.0] * 12
        b = [1.0] * 12
        result = stats.permutation_test(a, b, n_permutations=500, seed=0)
        assert 0.0 < result["p_value"] <= 1.0

    def test_permutation_detects_no_effect(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = stats.permutation_test(a, list(a), n_permutations=1000, seed=0)
        assert result["p_value"] > 0.5

    def test_empty_samples_raise_rather_than_return_zero(self):
        with pytest.raises(ValueError):
            stats.bootstrap_ci([])


# ---------------------------------------------------------------------------
# Chunking invariants
# ---------------------------------------------------------------------------


class TestChunking:
    def test_stride_larger_than_window_is_rejected(self):
        with pytest.raises(ValueError, match="stride_tokens"):
            chunk_document("a b c", window_tokens=4, stride_tokens=8)

    def test_overlap_matches_the_documented_ratio(self):
        text = " ".join(f"w{i}" for i in range(2000))
        chunks = chunk_document(text, window_tokens=512, stride_tokens=64)
        assert len(chunks) > 1
        assert chunks[1].start_token - chunks[0].start_token == 64

    def test_whole_document_is_covered(self):
        text = " ".join(f"w{i}" for i in range(1000))
        chunks = chunk_document(text, window_tokens=100, stride_tokens=50)
        covered = set()
        for c in chunks:
            covered.update(range(c.start_token, c.start_token + c.token_count))
        assert covered == set(range(1000))

    def test_empty_document_yields_no_chunks(self):
        assert chunk_document("") == []

    def test_coverage_stats_detect_full_retention(self):
        chunks = [Chunk(i, f"word{i} common text", 3, i * 3) for i in range(5)]
        cov = coverage_stats(chunks, chunks)
        assert cov["vocabulary_retained_pct"] == 100.0
        assert cov["n_document_positions_lost"] == 0
