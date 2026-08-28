"""Document chunking and TF-IDF chunk filtering.

The filtering algorithm is preserved from the original ``optimization/chunk_filter.py``
because it works and its effect is real (58 -> 12 chunks on the AAPL 8-K). What
changes is measurement and honesty around it.

Audit findings addressed:

* **C3 — the filter was double-counted.** The serial baseline B1 applies the same
  filter with the same cap, so the filter cannot explain any speedup of the
  distributed pipeline *over B1*. Filtering is therefore modelled here as an
  independent factor that either arm may have, which is what makes the 2x2
  ablation possible.
* **M4 — the cost was misreported.** The paper states "~80 ms CPU, essentially
  free"; the measured cost in ``results/aapl.json`` was 12,341 ms. :func:`filter_chunks`
  returns its own wall-clock time so the number in the paper comes from a
  measurement rather than an assumption.
* **Quality was never measured.** Discarding 79-90% of chunks must cost
  something. :func:`coverage_stats` quantifies what is lost in token and
  vocabulary terms, which is what the quality/cost curve needs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    token_count: int
    start_token: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_token": self.start_token,
        }


@dataclass
class FilterResult:
    chunks: list[Chunk]
    n_before: int
    n_after: int
    reduction_pct: float
    kept_indices: list[int]
    elapsed_ms: float
    method: str
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "reduction_pct": self.reduction_pct,
            "kept_indices": self.kept_indices,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "method": self.method,
            "coverage": self.coverage,
        }


Tokenizer = Callable[[str], list[str]]


def whitespace_tokenizer(text: str) -> list[str]:
    """Fallback tokenizer.

    Chunk boundaries differ from the model's subword tokenizer, so token counts
    are approximate. Recorded in the result as ``tokenizer='whitespace'`` so a
    reader knows which was used; use :func:`make_hf_tokenizer` for boundaries
    that match the summariser's vocabulary.
    """
    return text.split()


def make_hf_tokenizer(checkpoint: str) -> Tokenizer:
    """Tokenizer matching a model's vocabulary, so chunks match its context window."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint)

    def _tokenize(text: str) -> list[str]:
        return tok.tokenize(text)

    return _tokenize


def chunk_document(
    text: str,
    *,
    window_tokens: int = 512,
    stride_tokens: int = 64,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Split text into overlapping windows.

    ``stride_tokens`` is the *step* between window starts, so the overlap is
    ``window_tokens - stride_tokens``. With the project's defaults (512 window,
    64 stride) adjacent windows share 87.5% of their tokens, which is precisely
    why near-duplicate filtering recovers so much: the redundancy is created by
    the chunking, not present in the document.
    """
    if stride_tokens >= window_tokens:
        raise ValueError(
            f"stride_tokens ({stride_tokens}) must be < window_tokens ({window_tokens})"
        )
    tok = tokenizer or whitespace_tokenizer
    tokens = tok(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(tokens):
        window = tokens[start : start + window_tokens]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=" ".join(window),
                token_count=len(window),
                start_token=start,
            )
        )
        chunk_id += 1
        if start + window_tokens >= len(tokens):
            break
        start += stride_tokens
    return chunks


def coverage_stats(
    original: Sequence[Chunk], kept: Sequence[Chunk]
) -> dict[str, Any]:
    """What the filter discarded, in token and vocabulary terms.

    Filtering is only "free" if the discarded chunks were genuinely redundant.
    Vocabulary coverage is the cheap proxy: if the kept chunks still contain
    nearly all distinct terms, the discarded ones were duplicates; if not, the
    filter is dropping content and the H2 quality claim is affected.
    """
    def vocab(chunks: Sequence[Chunk]) -> set[str]:
        v: set[str] = set()
        for c in chunks:
            v.update(w.lower() for w in c.text.split())
        return v

    orig_vocab = vocab(original)
    kept_vocab = vocab(kept)
    orig_tokens = sum(c.token_count for c in original)
    kept_tokens = sum(c.token_count for c in kept)

    # Distinct source positions covered: with overlapping windows, token counts
    # overstate content, so this measures how much of the document is reachable.
    def covered_positions(chunks: Sequence[Chunk]) -> set[int]:
        positions: set[int] = set()
        for c in chunks:
            positions.update(range(c.start_token, c.start_token + c.token_count))
        return positions

    orig_positions = covered_positions(original)
    kept_positions = covered_positions(kept)

    return {
        "vocabulary_retained_pct": (
            round(100 * len(kept_vocab) / len(orig_vocab), 2) if orig_vocab else None
        ),
        "n_vocabulary_terms_lost": len(orig_vocab) - len(kept_vocab),
        "tokens_retained_pct": (
            round(100 * kept_tokens / orig_tokens, 2) if orig_tokens else None
        ),
        "document_positions_retained_pct": (
            round(100 * len(kept_positions) / len(orig_positions), 2)
            if orig_positions
            else None
        ),
        "n_document_positions_lost": len(orig_positions) - len(kept_positions),
    }


class ChunkFilter:
    """TF-IDF information-density selection with greedy near-duplicate removal.

    Algorithm unchanged from the original implementation:
      1. TF-IDF vectorise all chunks.
      2. Score each chunk by summed TF-IDF weight (information density).
      3. Greedily take the highest-scoring chunk whose cosine similarity to every
         already-selected chunk is below ``sim_threshold``.
      4. Restore document order.
    """

    def __init__(
        self,
        *,
        sim_threshold: float = 0.85,
        max_chunks_8k: int = 12,
        max_chunks_10k: int = 20,
        tfidf_max_features: int = 512,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.max_chunks_8k = max_chunks_8k
        self.max_chunks_10k = max_chunks_10k
        self.tfidf_max_features = tfidf_max_features

    def cap_for(self, filing_type: str) -> int:
        return self.max_chunks_10k if filing_type in ("10-K", "10-Q") else self.max_chunks_8k

    def filter(
        self,
        chunks: Sequence[Chunk],
        *,
        filing_type: str = "8-K",
        max_chunks: int | None = None,
    ) -> FilterResult:
        t0 = time.perf_counter()
        n_before = len(chunks)
        cap = max_chunks if max_chunks is not None else self.cap_for(filing_type)

        if n_before <= cap:
            elapsed = (time.perf_counter() - t0) * 1000
            return FilterResult(
                chunks=list(chunks),
                n_before=n_before,
                n_after=n_before,
                reduction_pct=0.0,
                kept_indices=[c.chunk_id for c in chunks],
                elapsed_ms=elapsed,
                method="passthrough (already under cap)",
                coverage=coverage_stats(chunks, chunks),
            )

        try:
            kept = self._tfidf_select(chunks, cap)
            method = f"tfidf-greedy-dedup(sim<{self.sim_threshold}, cap={cap})"
        except ImportError as exc:
            log.warning("ChunkFilter: sklearn unavailable (%s); uniform sampling", exc)
            step = max(1, n_before // cap)
            kept = list(chunks)[::step][:cap]
            method = f"uniform-sample(cap={cap}) [sklearn unavailable]"

        elapsed = (time.perf_counter() - t0) * 1000
        return FilterResult(
            chunks=kept,
            n_before=n_before,
            n_after=len(kept),
            reduction_pct=round((n_before - len(kept)) / n_before * 100, 2),
            kept_indices=[c.chunk_id for c in kept],
            elapsed_ms=elapsed,
            method=method,
            coverage=coverage_stats(chunks, kept),
        )

    def _tfidf_select(self, chunks: Sequence[Chunk], cap: int) -> list[Chunk]:
        import numpy as np
        from scipy.sparse import vstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [c.text for c in chunks]
        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=1)).ravel()

        order = np.argsort(scores)[::-1]
        kept_idx: list[int] = []
        kept_vecs: list[Any] = []
        for idx in order:
            if len(kept_idx) >= cap:
                break
            vec = matrix[idx]
            if kept_vecs:
                max_sim = float(cosine_similarity(vec, vstack(kept_vecs)).max())
                if max_sim >= self.sim_threshold:
                    continue
            kept_idx.append(int(idx))
            kept_vecs.append(vec)

        kept_idx.sort()
        return [chunks[i] for i in kept_idx]
