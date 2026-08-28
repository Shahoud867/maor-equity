"""Multi-dimensional financial sentiment.

Three audit findings are corrected here.

**M1 — "3-D sentiment" was two checkpoints, not three.** The temporal dimension
reused the market pipeline object (``self._pipe_tmp = self._pipe_mkt``), so it was
the same model applied to a keyword-selected subset of the same chunks. Measuring
this (``docs/AUDIT_RESPONSE.md``, finding N3) showed it was not merely "the same
model" in spirit but literally identical output: 985 of 985 routed temporal
labels matched their market label exactly, because deterministic inference on
identical weights given identical text cannot do otherwise. Declaring the sharing
(as a first pass) made the defect visible; it did not fix it.

The fix here is a genuinely independent third checkpoint:
``yiyanghkust/finbert-fls``, a BERT model fine-tuned specifically to classify
forward-looking statements (Not FLS / Non-specific FLS / Specific FLS) — the
semantically correct model for "temporal" content, and a different label space
from market/regulatory's positive/neutral/negative. That heterogeneity is real:
a specific forward commitment is not "positive" in the way a strong quarter is,
so :class:`DimensionSpec` now carries its own ``label_scheme`` and the sentiment
matrix stores per-row label orders rather than forcing every dimension onto one
scheme.

**M2 — the regulatory row was computed from a placeholder sentence.** When no
chunk matched the regulatory regex, the router emitted the literal string
"No regulatory content detected." and classified *that*, producing a confident
neutral row that was published as a real measurement. Dimensions with no matching
content are now :class:`DimensionResult` with ``present=False`` and no scores,
and the aggregate vector marks the row as missing rather than inventing one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Polarity labels: market and regulatory sentiment.
LABELS = ("positive", "neutral", "negative")

# Forward-looking-statement specificity labels: temporal sentiment. Not a
# polarity scale — "specific_fls" means a concrete forward commitment was made,
# not that the commitment was optimistic. Order matches finbert-fls's own
# id2label (0=Not FLS, 1=Non-specific FLS, 2=Specific FLS).
FLS_LABELS = ("not_fls", "nonspecific_fls", "specific_fls")

LABEL_SCHEMES: dict[str, tuple[str, ...]] = {"polarity": LABELS, "fls": FLS_LABELS}

_FLS_RAW_TO_CANONICAL = {
    "notfls": "not_fls",
    "nonspecificfls": "nonspecific_fls",
    "specificfls": "specific_fls",
}


def normalise_label(raw: str, scheme: str) -> str:
    """Map a pipeline's raw label string onto the canonical key for its scheme."""
    key = raw.strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    if scheme == "polarity":
        if key not in LABELS:
            raise ValueError(f"unrecognised polarity label {raw!r}")
        return key
    if scheme == "fls":
        if key not in _FLS_RAW_TO_CANONICAL:
            raise ValueError(f"unrecognised FLS label {raw!r}")
        return _FLS_RAW_TO_CANONICAL[key]
    raise ValueError(f"unknown label scheme {scheme!r}")


# Ordinary financial English. If a tokenizer cannot resolve most of these to real
# vocabulary entries, it is misconfigured, whatever it reports about itself.
_TOKENIZER_PROBE = (
    "Revenue grew twenty percent year over year .",
    "The company reported a net loss for the quarter .",
    "Regulatory compliance costs increased significantly .",
)


class TokenizerVerificationError(RuntimeError):
    """Raised when a tokenizer loads but does not actually tokenize."""


def verify_tokenizer(tokenizer: Any, checkpoint: str, *, max_unk_ratio: float = 0.30) -> None:
    """Fail loudly if a tokenizer maps ordinary text to unknown tokens.

    This guards a silent-corruption failure mode rather than a crash. A
    misconfigured WordPiece vocabulary still returns tensors, the model still
    returns probabilities, and the probabilities are constant and confident —
    which is exactly what a plausible-but-meaningless result looks like. It was
    observed for ``yiyanghkust/finbert-tone`` under transformers 5.x, where every
    content word became ``[UNK]`` and every sentence scored Neutral at p=1.00.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        return

    total = 0
    unknown = 0
    for probe in _TOKENIZER_PROBE:
        ids = tokenizer(probe)["input_ids"]
        special = set(tokenizer.all_special_ids or [])
        content = [i for i in ids if i not in special or i == unk_id]
        total += len(content)
        unknown += sum(1 for i in content if i == unk_id)

    if total == 0:
        raise TokenizerVerificationError(
            f"{checkpoint}: tokenizer produced no content tokens for probe text"
        )
    ratio = unknown / total
    if ratio > max_unk_ratio:
        raise TokenizerVerificationError(
            f"{checkpoint}: tokenizer mapped {ratio:.0%} of ordinary financial "
            f"English to [UNK] (threshold {max_unk_ratio:.0%}). The vocabulary is "
            f"not being applied correctly. Any scores from this model would be "
            f"constant and meaningless rather than obviously broken, so this is a "
            f"hard error."
        )


@dataclass(frozen=True)
class DimensionSpec:
    """A sentiment dimension, the checkpoint that scores it, and its label space.

    ``label_scheme`` is looked up in :data:`LABEL_SCHEMES`. Dimensions are not
    required to share a label space — market/regulatory are polarity, temporal
    is forward-looking-statement specificity — and forcing them onto one scheme
    is what let a reused checkpoint pass as an "independent" dimension.
    """

    name: str
    checkpoint: str
    label_scheme: str = "polarity"
    # When set, this dimension is scored by the same loaded weights as another
    # dimension. Recorded so the distinction survives into the paper. None of
    # the three default dimensions share a checkpoint any more.
    shares_checkpoint_with: str | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return LABEL_SCHEMES[self.label_scheme]


@dataclass
class DimensionResult:
    """Scores for one dimension over the chunks routed to it.

    ``present=False`` means no content was routed here. It does not mean neutral
    — and for the ``fls`` scheme there is no "neutral" to fall back to at all.
    """

    dimension: str
    present: bool
    checkpoint: str
    label_scheme: str = "polarity"
    shares_checkpoint_with: str | None = None
    scores: list[dict[str, float]] = field(default_factory=list)
    n_texts: int = 0
    n_low_confidence: int = 0
    mean_vector: list[float] | None = None

    @property
    def label_order(self) -> tuple[str, ...]:
        return LABEL_SCHEMES[self.label_scheme]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "present": self.present,
            "checkpoint": self.checkpoint,
            "label_scheme": self.label_scheme,
            "shares_checkpoint_with": self.shares_checkpoint_with,
            "n_texts": self.n_texts,
            "n_low_confidence": self.n_low_confidence,
            "mean_vector": self.mean_vector,
            "label_order": list(self.label_order),
        }


@dataclass
class SentimentMatrix:
    """The 3xN sentiment matrix with explicit missingness and per-row label spaces.

    ``rows`` is ordered as ``dimension_order``. A missing dimension is a row of
    NaN and a False entry in :attr:`present`, so a consumer cannot silently treat
    "absent" as "neutral". ``label_schemes`` is parallel to ``dimension_order``:
    rows are not assumed to share a label space, because they no longer do.
    """

    rows: np.ndarray
    dimension_order: tuple[str, ...]
    present: tuple[bool, ...]
    label_schemes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label_schemes:
            # Backward-compatible default for callers that only ever used
            # polarity dimensions (e.g. ad-hoc tests).
            self.label_schemes = tuple("polarity" for _ in self.dimension_order)

    @property
    def n_present(self) -> int:
        return sum(self.present)

    def label_order_for(self, dimension: str) -> tuple[str, ...]:
        i = self.dimension_order.index(dimension)
        return LABEL_SCHEMES[self.label_schemes[i]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [
                [None if np.isnan(v) else round(float(v), 6) for v in row]
                for row in self.rows
            ],
            "dimension_order": list(self.dimension_order),
            "label_schemes": list(self.label_schemes),
            "per_dimension_labels": {
                dim: list(LABEL_SCHEMES[scheme])
                for dim, scheme in zip(self.dimension_order, self.label_schemes)
            },
            "present": list(self.present),
            "n_present_dimensions": self.n_present,
        }

    def direction(self, dimension: str) -> str | None:
        """argmax label for one dimension, in that dimension's own label space."""
        if dimension not in self.dimension_order:
            raise KeyError(dimension)
        i = self.dimension_order.index(dimension)
        if not self.present[i]:
            return None
        labels = LABEL_SCHEMES[self.label_schemes[i]]
        return labels[int(np.nanargmax(self.rows[i]))]


class DimensionRouter:
    """Routes chunks to dimensions by keyword match.

    This is a lexical heuristic, not a learned router, and the paper must say so.
    :meth:`route` returns coverage statistics so routing quality is measurable
    rather than assumed — the audit noted the original had no precision/recall
    evaluation of any kind.
    """

    REGULATORY_PATTERNS = (
        r"\bSEC\b", r"\blitigation\b", r"\bcompliance\b", r"\bpenalt",
        r"\benforcement\b", r"\brestatement\b", r"\bauditor\b",
        r"\bregulat", r"\bFDA\b", r"\bFTC\b", r"\bsubpoena\b",
        r"\binvestigation\b", r"\bsettlement\b", r"\bconsent decree\b",
    )
    TEMPORAL_PATTERNS = (
        r"\bwill\b", r"\bexpect", r"\banticipat", r"\bnext quarter\b",
        r"\bforecast", r"\bguidance\b", r"\boutlook\b", r"\bFY\d{4}\b",
        r"\bgoing forward\b", r"\bfuture\b", r"\bproject(?:ed|ions)\b",
    )

    def __init__(
        self,
        regulatory_patterns: Sequence[str] | None = None,
        temporal_patterns: Sequence[str] | None = None,
    ) -> None:
        self._reg = [
            re.compile(p, re.I) for p in (regulatory_patterns or self.REGULATORY_PATTERNS)
        ]
        self._tmp = [
            re.compile(p, re.I) for p in (temporal_patterns or self.TEMPORAL_PATTERNS)
        ]

    def route(self, chunks: Iterable[Any]) -> dict[str, Any]:
        """Route chunk texts to dimensions.

        Accepts chunk dicts with a ``text`` key, or plain strings.
        Absent dimensions get an empty list — never a placeholder sentence.
        """
        texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
        market = list(texts)
        regulatory = [t for t in texts if any(p.search(t) for p in self._reg)]
        temporal = [t for t in texts if any(p.search(t) for p in self._tmp)]

        n = len(texts) or 1
        return {
            "routed": {
                "market": market,
                "regulatory": regulatory,
                "temporal": temporal,
            },
            "coverage": {
                "n_chunks": len(texts),
                "market_pct": 100.0,
                "regulatory_pct": round(len(regulatory) / n * 100, 2),
                "temporal_pct": round(len(temporal) / n * 100, 2),
                "regulatory_absent": not regulatory,
                "temporal_absent": not temporal,
            },
            "router_type": "lexical-regex",
            "n_regulatory_patterns": len(self._reg),
            "n_temporal_patterns": len(self._tmp),
        }


class SentimentBundle:
    """Loads the sentiment checkpoints once and scores each dimension.

    Device and quantisation are configurable so the identical code path runs on a
    CUDA node and on CPU. The CPU path is what makes the H3 evaluation runnable
    without the GPU node.
    """

    def __init__(
        self,
        market_checkpoint: str = "ProsusAI/finbert",
        regulatory_checkpoint: str = "yiyanghkust/finbert-tone",
        temporal_checkpoint: str = "yiyanghkust/finbert-fls",
        *,
        device: str = "cpu",
        quantisation: str = "none",
        min_confidence: float = 0.60,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.quantisation = quantisation

        # Three genuinely distinct checkpoints. Temporal previously reused the
        # market pipeline object; measuring that (985/985 identical labels on
        # PhraseBank) showed it produced no independent signal at all. finbert-fls
        # is purpose-built for forward-looking-statement detection, which is what
        # "temporal" was always supposed to mean, and its label space
        # (specificity, not polarity) is genuinely different from market's.
        self.specs = (
            DimensionSpec("market", market_checkpoint, label_scheme="polarity"),
            DimensionSpec("regulatory", regulatory_checkpoint, label_scheme="polarity"),
            DimensionSpec("temporal", temporal_checkpoint, label_scheme="fls"),
        )
        self.dimension_order = tuple(s.name for s in self.specs)

        self._pipelines: dict[str, Any] = {}
        self._loaded_checkpoints: dict[str, Any] = {}

    # -- loading ---------------------------------------------------------

    @staticmethod
    def _load_legacy_bert(checkpoint: str) -> tuple[Any, Any]:
        """Load a pre-`model_type` BERT checkpoint (e.g. yiyanghkust/finbert-tone).

        That repository predates current config conventions: its ``config.json``
        has no ``model_type`` key and it ships only ``vocab.txt`` — no
        ``tokenizer_config.json``, no ``tokenizer.json``. Under transformers 5.x
        this breaks in two separate ways, and the second is dangerous:

        1. ``AutoConfig``/``AutoTokenizer`` raise outright on the missing
           ``model_type``.
        2. ``BertTokenizerFast(vocab_file=...)`` *appears* to work but maps every
           content word to ``[UNK]``, because this vocab uses a non-standard
           special-token layout ([PAD], [EOS], [UNK], [CLS], [SEP], [MASK] at
           indices 0-5 rather than BERT's). The model then returns a confident,
           constant label for every input — plausible-looking numbers that mean
           nothing.

        Failure mode (2) is silent, so :func:`verify_tokenizer` is called on the
        result and raises if the vocabulary is not actually being hit.
        """
        import json as _json

        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
        from tokenizers.models import WordPiece
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            PreTrainedTokenizerFast,
        )

        vocab_path = hf_hub_download(checkpoint, "vocab.txt")
        vocab = {
            line: idx
            for idx, line in enumerate(
                Path(vocab_path).read_text(encoding="utf-8").splitlines()
            )
        }
        for required in ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"):
            if required not in vocab:
                raise RuntimeError(
                    f"{checkpoint}: vocab.txt is missing the {required} token; "
                    f"cannot build a WordPiece tokenizer for it"
                )

        backend = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
        backend.normalizer = normalizers.BertNormalizer(lowercase=True)
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
        )
        backend.decoder = decoders.WordPiece(prefix="##")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            model_max_length=512,
        )
        verify_tokenizer(tokenizer, checkpoint)

        config_path = hf_hub_download(checkpoint, "config.json")
        raw_config = _json.loads(Path(config_path).read_text(encoding="utf-8"))
        raw_config.setdefault("model_type", "bert")
        model = BertForSequenceClassification.from_pretrained(
            checkpoint, config=BertConfig(**raw_config)
        )
        log.info("sentiment: loaded %s via legacy-BERT compatibility path", checkpoint)
        return tokenizer, model

    def load(self) -> "SentimentBundle":
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        quant_kwargs: dict[str, Any] = {}
        if self.quantisation != "none":
            if self.device == "cpu":
                raise ValueError(
                    f"quantisation={self.quantisation!r} requires CUDA; bitsandbytes "
                    f"does not support CPU inference for this path. Use "
                    f"quantisation='none' on CPU."
                )
            import torch
            from transformers import BitsAndBytesConfig

            if self.quantisation == "nf4":
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self.quantisation == "int8":
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )

        for spec in self.specs:
            if spec.checkpoint in self._loaded_checkpoints:
                # Reuse: this is the VRAM saving, and it is why temporal is not
                # an independent dimension.
                self._pipelines[spec.name] = self._loaded_checkpoints[spec.checkpoint]
                log.info(
                    "sentiment[%s]: reusing already-loaded checkpoint %s",
                    spec.name,
                    spec.checkpoint,
                )
                continue

            model_kwargs = dict(quant_kwargs)
            if self.device != "cpu" and not quant_kwargs:
                model_kwargs["device_map"] = (
                    {"": int(self.device.split(":")[-1])} if ":" in self.device else {"": 0}
                )
            try:
                tok = AutoTokenizer.from_pretrained(spec.checkpoint)
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    spec.checkpoint, **model_kwargs
                )
                verify_tokenizer(tok, spec.checkpoint)
            except Exception as exc:
                log.warning(
                    "sentiment[%s]: standard loading of %s failed (%s); "
                    "trying legacy-BERT compatibility path",
                    spec.name,
                    spec.checkpoint,
                    type(exc).__name__,
                )
                tok, mdl = self._load_legacy_bert(spec.checkpoint)
                if model_kwargs.get("device_map"):
                    mdl = mdl.to(f"cuda:{self.device.split(':')[-1]}" if ":" in self.device else "cuda:0")

            pipe_kwargs: dict[str, Any] = {"top_k": None}
            if self.device == "cpu":
                pipe_kwargs["device"] = -1
            pipe = pipeline(
                "text-classification", model=mdl, tokenizer=tok, **pipe_kwargs
            )
            self._loaded_checkpoints[spec.checkpoint] = pipe
            self._pipelines[spec.name] = pipe
            log.info("sentiment[%s]: loaded %s", spec.name, spec.checkpoint)

        return self

    @property
    def n_distinct_checkpoints(self) -> int:
        """How many sets of weights are actually resident. Reported in results."""
        return len(self._loaded_checkpoints)

    # -- scoring ---------------------------------------------------------

    def classify(self, texts: Sequence[str], dimension: str) -> DimensionResult:
        spec = next(s for s in self.specs if s.name == dimension)
        labels = spec.labels
        if not texts:
            # The M2 fix: absent, not neutral.
            return DimensionResult(
                dimension=dimension,
                present=False,
                checkpoint=spec.checkpoint,
                label_scheme=spec.label_scheme,
                shares_checkpoint_with=spec.shares_checkpoint_with,
                n_texts=0,
                mean_vector=None,
            )

        pipe = self._pipelines.get(dimension)
        if pipe is None:
            raise RuntimeError(f"dimension {dimension!r} not loaded; call load() first")

        raw = pipe(
            list(texts),
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )
        scores: list[dict[str, float]] = []
        n_low = 0
        for item in raw:
            dist = {
                normalise_label(s["label"], spec.label_scheme): float(s["score"])
                for s in item
            }
            normalised = {lab: dist.get(lab, 0.0) for lab in labels}
            if max(normalised.values()) < self.min_confidence:
                n_low += 1
            scores.append(normalised)

        arr = np.array([[s[lab] for lab in labels] for s in scores], dtype=float)
        mean_vec = arr.mean(axis=0)
        total = mean_vec.sum()
        if total > 0:
            mean_vec = mean_vec / total

        return DimensionResult(
            dimension=dimension,
            present=True,
            checkpoint=spec.checkpoint,
            label_scheme=spec.label_scheme,
            shares_checkpoint_with=spec.shares_checkpoint_with,
            scores=scores,
            n_texts=len(texts),
            n_low_confidence=n_low,
            mean_vector=[round(float(v), 6) for v in mean_vec],
        )

    def classify_all(self, routed: dict[str, list[str]]) -> dict[str, DimensionResult]:
        return {
            spec.name: self.classify(routed.get(spec.name, []), spec.name)
            for spec in self.specs
        }

    def unload(self, *, strict: bool = False) -> Any:
        """Release every loaded checkpoint and verify the memory came back.

        A HuggingFace ``pipeline`` holds both the model and the tokenizer, so
        clearing the dictionaries is not enough on its own — the underlying
        modules are moved off the device first, while the references are still
        valid.
        """
        from ..gpu.lifecycle import ModelRegistry, release_torch_module

        pipelines = list(self._pipelines.values())
        checkpoints = list(self._loaded_checkpoints.items())
        self._pipelines.clear()
        self._loaded_checkpoints.clear()

        if not pipelines and not checkpoints:
            return None

        device_index = 0 if self.device == "cpu" else int(str(self.device).split(":")[-1] or 0)
        registry = ModelRegistry.instance()
        for checkpoint, _pipe in checkpoints:
            registry.unregister(checkpoint, device_index)

        objects: list[Any] = list(pipelines)
        for _checkpoint, pipe in checkpoints:
            objects.append(pipe)
            objects.append(getattr(pipe, "model", None))
            objects.append(getattr(pipe, "tokenizer", None))

        verification = release_torch_module(
            "sentiment-bundle",
            *objects,
            device=device_index,
            strict=strict,
        )
        log.info(
            "unloaded sentiment bundle (%d checkpoint(s)): freed %.0f MB, "
            "%.0f MB still held",
            len(checkpoints),
            verification.allocated_freed_mb,
            verification.residual_mb,
        )
        return verification

    def __enter__(self) -> "SentimentBundle":
        return self.load()

    def __exit__(self, *exc: object) -> bool:
        self.unload()
        return False


def build_matrix(results: dict[str, DimensionResult]) -> SentimentMatrix:
    """Assemble the sentiment matrix, marking absent dimensions as NaN.

    Rows may use different label schemes (market/regulatory are polarity,
    temporal is FLS-specificity); ``label_schemes`` on the returned matrix
    records which scheme applies to each row so a consumer never reads a
    temporal cell as if it were a positive/negative score.
    """
    order = tuple(results.keys())
    rows = []
    present = []
    schemes = []
    for name in order:
        r = results[name]
        schemes.append(r.label_scheme)
        if r.present and r.mean_vector is not None:
            rows.append(r.mean_vector)
            present.append(True)
        else:
            rows.append([np.nan] * len(r.label_order))
            present.append(False)
    return SentimentMatrix(
        rows=np.array(rows, dtype=float),
        dimension_order=order,
        present=tuple(present),
        label_schemes=tuple(schemes),
    )
