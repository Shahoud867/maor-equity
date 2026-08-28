"""Multi-dimensional financial sentiment.

Two audit findings are corrected here, and one piece of honesty is enforced in
the code so it cannot drift back out of the paper.

**M1 — "3-D sentiment" was two checkpoints, not three.** The temporal dimension
reused the market pipeline object (``self._pipe_tmp = self._pipe_mkt``), so it was
the same model applied to a keyword-selected subset of the same chunks. That is
not a third dimension. The sharing is legitimate on a 4 GB card and is kept, but
it is now named (:attr:`DimensionSpec.shares_checkpoint_with`) and surfaced in
every result, so no downstream text can describe this as three independent models.

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

# Canonical label order for every sentiment row.
LABELS = ("positive", "neutral", "negative")


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
    """A sentiment dimension and the checkpoint that scores it."""

    name: str
    checkpoint: str
    # When set, this dimension is scored by the same loaded weights as another
    # dimension. Recorded so the distinction survives into the paper.
    shares_checkpoint_with: str | None = None


@dataclass
class DimensionResult:
    """Scores for one dimension over the chunks routed to it.

    ``present=False`` means no content was routed here. It does not mean neutral.
    """

    dimension: str
    present: bool
    checkpoint: str
    shares_checkpoint_with: str | None = None
    scores: list[dict[str, float]] = field(default_factory=list)
    n_texts: int = 0
    n_low_confidence: int = 0
    mean_vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "present": self.present,
            "checkpoint": self.checkpoint,
            "shares_checkpoint_with": self.shares_checkpoint_with,
            "n_texts": self.n_texts,
            "n_low_confidence": self.n_low_confidence,
            "mean_vector": self.mean_vector,
            "label_order": list(LABELS),
        }


@dataclass
class SentimentMatrix:
    """The 3xN sentiment matrix with explicit missingness.

    ``rows`` is ordered as ``dimension_order``. A missing dimension is a row of
    NaN and a False entry in :attr:`present`, so a consumer cannot silently treat
    "absent" as "neutral".
    """

    rows: np.ndarray
    dimension_order: tuple[str, ...]
    present: tuple[bool, ...]
    label_order: tuple[str, ...] = LABELS

    @property
    def n_present(self) -> int:
        return sum(self.present)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [
                [None if np.isnan(v) else round(float(v), 6) for v in row]
                for row in self.rows
            ],
            "dimension_order": list(self.dimension_order),
            "label_order": list(self.label_order),
            "present": list(self.present),
            "n_present_dimensions": self.n_present,
        }

    def direction(self, dimension: str) -> str | None:
        """argmax label for one dimension, or None when the dimension is absent."""
        if dimension not in self.dimension_order:
            raise KeyError(dimension)
        i = self.dimension_order.index(dimension)
        if not self.present[i]:
            return None
        return self.label_order[int(np.nanargmax(self.rows[i]))]


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

        # Temporal deliberately shares the market checkpoint. Declared, not hidden.
        self.specs = (
            DimensionSpec("market", market_checkpoint),
            DimensionSpec("regulatory", regulatory_checkpoint),
            DimensionSpec(
                "temporal", market_checkpoint, shares_checkpoint_with="market"
            ),
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
        if not texts:
            # The M2 fix: absent, not neutral.
            return DimensionResult(
                dimension=dimension,
                present=False,
                checkpoint=spec.checkpoint,
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
            dist = {s["label"].lower(): float(s["score"]) for s in item}
            # finbert-tone emits Positive/Negative/Neutral; ProsusAI emits
            # lowercase. Normalising here keeps the label order canonical.
            normalised = {lab: dist.get(lab, 0.0) for lab in LABELS}
            if max(normalised.values()) < self.min_confidence:
                n_low += 1
            scores.append(normalised)

        arr = np.array([[s[lab] for lab in LABELS] for s in scores], dtype=float)
        mean_vec = arr.mean(axis=0)
        total = mean_vec.sum()
        if total > 0:
            mean_vec = mean_vec / total

        return DimensionResult(
            dimension=dimension,
            present=True,
            checkpoint=spec.checkpoint,
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

    def unload(self) -> None:
        self._pipelines.clear()
        self._loaded_checkpoints.clear()
        from ..hardware import release_vram

        release_vram()


def build_matrix(results: dict[str, DimensionResult]) -> SentimentMatrix:
    """Assemble the sentiment matrix, marking absent dimensions as NaN."""
    order = tuple(results.keys())
    rows = []
    present = []
    for name in order:
        r = results[name]
        if r.present and r.mean_vector is not None:
            rows.append(r.mean_vector)
            present.append(True)
        else:
            rows.append([np.nan] * len(LABELS))
            present.append(False)
    return SentimentMatrix(
        rows=np.array(rows, dtype=float),
        dimension_order=order,
        present=tuple(present),
    )
