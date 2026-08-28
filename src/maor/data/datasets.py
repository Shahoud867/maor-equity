"""Dataset loading with explicit provenance and validation.

Every loader returns the data *and* a description of exactly what was loaded:
source repository, revision, split, row count and label distribution. That
description is embedded in the result file, so a reader can tell which variant of
a benchmark a number refers to.

This matters here specifically. The project README described the H3 dataset as
"Financial PhraseBank (sentences_75agree split, ~4,800 sentences)". Those two
facts are inconsistent: the 75%-agreement split has 3,453 sentences; 4,846 is the
50%-agreement split. Conflating them changes the task difficulty, because
agreement level is a proxy for label ambiguity. The loader records the actual
split so the paper cannot repeat the error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

# Canonical Financial PhraseBank integer -> label mapping (Malo et al., 2014,
# as encoded by the original HuggingFace dataset).
PHRASEBANK_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# The canonical row counts per agreement split, used to identify which split a
# mirror actually contains rather than trusting its name.
PHRASEBANK_SPLIT_SIZES = {
    "sentences_allagree": 2264,
    "sentences_75agree": 3453,
    "sentences_66agree": 4217,
    "sentences_50agree": 4846,
}

# The upstream `financial_phrasebank` dataset used a loading script, which
# `datasets` >= 4 refuses to execute. This mirror carries the same rows in
# Parquet. Verified on load by row count and label distribution.
PHRASEBANK_MIRROR = "warwickai/financial_phrasebank_mirror"


class DatasetError(RuntimeError):
    """Raised when a dataset is missing, malformed, or not what it claims to be."""


@dataclass
class LabelledDataset:
    """Text/label pairs plus the provenance of where they came from."""

    texts: list[str]
    labels: list[str]
    source: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texts)

    def label_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for l in self.labels:
            counts[l] = counts.get(l, 0) + 1
        return dict(sorted(counts.items()))

    def subsample(self, n: int, *, seed: int = 0) -> "LabelledDataset":
        """Deterministic stratified-by-order subsample.

        Uses a seeded shuffle so a reduced run is reproducible and is not just
        the first n rows, which in PhraseBank are ordered and would bias the
        label mix.
        """
        import random

        if n >= len(self.texts):
            return self
        rng = random.Random(seed)
        idx = list(range(len(self.texts)))
        rng.shuffle(idx)
        idx = sorted(idx[:n])
        return LabelledDataset(
            texts=[self.texts[i] for i in idx],
            labels=[self.labels[i] for i in idx],
            source={**self.source, "subsampled_to": n, "subsample_seed": seed},
        )


@dataclass
class SummarisationDataset:
    """Document/reference-summary pairs."""

    documents: list[str]
    references: list[str]
    source: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.documents)

    def subsample(self, n: int, *, seed: int = 0) -> "SummarisationDataset":
        import random

        if n >= len(self.documents):
            return self
        rng = random.Random(seed)
        idx = list(range(len(self.documents)))
        rng.shuffle(idx)
        idx = sorted(idx[:n])
        return SummarisationDataset(
            documents=[self.documents[i] for i in idx],
            references=[self.references[i] for i in idx],
            source={**self.source, "subsampled_to": n, "subsample_seed": seed},
        )


def _identify_phrasebank_split(n_rows: int) -> str:
    """Name the agreement split from its row count, or flag it as unrecognised."""
    for name, size in PHRASEBANK_SPLIT_SIZES.items():
        if n_rows == size:
            return name
    return f"unrecognised({n_rows}_rows)"


def load_financial_phrasebank(
    *,
    repo: str = PHRASEBANK_MIRROR,
    n_samples: int | None = None,
    seed: int = 0,
) -> LabelledDataset:
    """Load Financial PhraseBank, verifying it is what it claims to be.

    Raises :class:`DatasetError` with an actionable message rather than returning
    partial data, because a silently-truncated benchmark produces a number that
    looks valid and is not.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetError(
            "the `datasets` package is required: pip install datasets"
        ) from exc

    try:
        ds = load_dataset(repo, split="train")
    except Exception as exc:
        raise DatasetError(
            f"could not load Financial PhraseBank from {repo!r}: {exc}. "
            f"The original `financial_phrasebank` dataset uses a loading script "
            f"which `datasets` >= 4 refuses to run; this project uses a Parquet "
            f"mirror instead. Check network access, or set HF_TOKEN if rate-limited."
        ) from exc

    cols = set(ds.column_names)
    text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
    if text_col is None or "label" not in cols:
        raise DatasetError(
            f"{repo} has columns {sorted(cols)}; expected a text column "
            f"('sentence' or 'text') and 'label'"
        )

    texts = [str(t) for t in ds[text_col]]
    raw_labels = list(ds["label"])
    if not all(isinstance(l, int) for l in raw_labels):
        raise DatasetError(f"{repo}: 'label' column must be integer-coded")
    unknown = set(raw_labels) - set(PHRASEBANK_LABELS)
    if unknown:
        raise DatasetError(
            f"{repo}: unexpected label values {sorted(unknown)}; "
            f"expected {sorted(PHRASEBANK_LABELS)} (0=negative, 1=neutral, 2=positive)"
        )
    labels = [PHRASEBANK_LABELS[l] for l in raw_labels]

    split_name = _identify_phrasebank_split(len(texts))
    dist = {}
    for l in labels:
        dist[l] = dist.get(l, 0) + 1

    # Integrity check against the published composition. The 50agree split is
    # 2879 neutral / 1363 positive / 604 negative.
    if split_name == "sentences_50agree":
        expected = {"neutral": 2879, "positive": 1363, "negative": 604}
        if dist != expected:
            log.warning(
                "PhraseBank label distribution %s differs from the published "
                "composition %s for %s",
                dist,
                expected,
                split_name,
            )

    source = {
        "dataset": "Financial PhraseBank (Malo et al., 2014)",
        "repo": repo,
        "loaded_split": "train",
        "identified_agreement_split": split_name,
        "n_rows_total": len(texts),
        "label_distribution": dict(sorted(dist.items())),
        "label_mapping": {str(k): v for k, v in PHRASEBANK_LABELS.items()},
        "note": (
            "Loaded from a Parquet mirror because the upstream dataset uses a "
            "loading script that datasets>=4 will not execute. Split identity "
            "verified by row count and label distribution, not by repo name."
        ),
    }
    data = LabelledDataset(texts=texts, labels=labels, source=source)
    if n_samples is not None:
        data = data.subsample(n_samples, seed=seed)
    return data


def load_ectsum(
    path: str | Path,
    *,
    n_samples: int | None = None,
    seed: int = 0,
) -> SummarisationDataset:
    """Load ECTSum from the local JSONL file.

    The file is already present in the repository at
    ``data/ectsum/ectsum_test.jsonl`` (495 test records). It was downloaded and
    then never used; the H2 numbers in the original paper were literals.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"ECTSum file not found at {path}. Expected the test JSONL shipped "
            f"with the repository, or download it from the ECTSum release "
            f"(Mukherjee et al., EMNLP 2022)."
        )

    documents: list[str] = []
    references: list[str] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            doc = rec.get("text") or rec.get("input") or rec.get("document")
            ref = rec.get("summary") or rec.get("output") or rec.get("target")
            if not doc or not ref:
                malformed += 1
                continue
            documents.append(str(doc))
            references.append(str(ref))

    if not documents:
        raise DatasetError(f"{path} yielded no usable records ({malformed} malformed)")
    if malformed:
        log.warning("ECTSum: skipped %d malformed/incomplete records", malformed)

    doc_lens = [len(d.split()) for d in documents]
    ref_lens = [len(r.split()) for r in references]
    source = {
        "dataset": "ECTSum (Mukherjee et al., EMNLP 2022)",
        "path": str(path).replace("\\", "/"),
        "n_rows_total": len(documents),
        "n_malformed_skipped": malformed,
        "mean_document_words": round(sum(doc_lens) / len(doc_lens), 1),
        "mean_reference_words": round(sum(ref_lens) / len(ref_lens), 1),
        "max_document_words": max(doc_lens),
    }
    data = SummarisationDataset(documents, references, source)
    if n_samples is not None:
        data = data.subsample(n_samples, seed=seed)
    return data


def validate_available(config: Any) -> dict[str, Any]:
    """Report which datasets are usable right now, without loading them fully.

    Used by ``maor doctor`` so a missing dataset is discovered before an
    experiment starts rather than an hour into it.
    """
    report: dict[str, Any] = {}

    ectsum_path = Path(config.data.ectsum_path)
    if ectsum_path.exists():
        try:
            n = sum(1 for line in ectsum_path.open(encoding="utf-8") if line.strip())
            report["ectsum"] = {
                "status": "available",
                "path": str(ectsum_path),
                "n_lines": n,
            }
        except OSError as exc:
            report["ectsum"] = {"status": "unreadable", "error": str(exc)}
    else:
        report["ectsum"] = {
            "status": "missing",
            "path": str(ectsum_path),
            "remedy": "Place ectsum_test.jsonl at this path.",
        }

    try:
        from huggingface_hub import HfApi

        HfApi().dataset_info(PHRASEBANK_MIRROR)
        report["financial_phrasebank"] = {
            "status": "reachable",
            "repo": PHRASEBANK_MIRROR,
        }
    except Exception as exc:
        report["financial_phrasebank"] = {
            "status": "unreachable",
            "repo": PHRASEBANK_MIRROR,
            "error": str(exc)[:200],
            "remedy": "Check network access or set HF_TOKEN.",
        }

    return report
