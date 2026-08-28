"""
optimization/chunk_filter.py — Node A (CPU)
TF-IDF based chunk deduplication to reduce Phi-3-mini inference load on Node B.

WHY THIS EXISTS (H1 latency fix):
  Phi-3-mini takes ~15s per chunk on a 4 GB T1000.
  A typical 8-K with 512-token windows at 64-stride overlap produces 15-25 chunks,
  but adjacent overlapping chunks share 87.5% of their tokens — most are near-duplicates.
  Filtering to 8-12 high-information, non-redundant chunks cuts map-step time by ~50%,
  which is the dominant term in the pipeline and directly enables the H1 30-50% target.

ALGORITHM:
  1. TF-IDF vectorize all chunks (CPU, ~0.1s for 25 chunks)
  2. Score each chunk by total TF-IDF weight (proxy for information density)
  3. Greedy selection: take highest-scoring chunk not too similar (cosine > threshold)
     to any already-selected chunk
  4. Re-sort selected chunks into original document order (preserve narrative flow)

TUNING:
  sim_threshold=0.85 — cosine similarity above which a chunk is considered redundant
  max_chunks=12      — hard cap on chunks sent to Node B
  Both are configurable per filing type (8-K needs fewer chunks than 10-K).
"""

import numpy as np


class ChunkFilter:
    """
    Runs on Node A CPU. Filters a list of chunk dicts (each with 'text', 'chunk_id')
    and returns a deduplicated subset preserving document order.
    """

    def __init__(self,
                 sim_threshold: float = 0.85,
                 max_chunks_8k: int = 12,
                 max_chunks_10k: int = 20):
        self.sim_threshold   = sim_threshold
        self.max_chunks_8k   = max_chunks_8k
        self.max_chunks_10k  = max_chunks_10k

    # ------------------------------------------------------------------
    def filter(self, chunks: list, filing_type: str = "8-K") -> dict:
        """
        Parameters
        ----------
        chunks      : list of {'chunk_id': int, 'text': str, 'token_count': int, ...}
        filing_type : '8-K' | '10-K' | '10-Q' — controls max_chunks cap

        Returns
        -------
        dict with keys:
            'chunks'           : filtered list (document order preserved)
            'n_before'         : original count
            'n_after'          : filtered count
            'reduction_pct'    : percentage of chunks dropped
            'kept_indices'     : original chunk_id values that were kept
        """
        n_before = len(chunks)
        max_cap  = self.max_chunks_10k if filing_type in ("10-K", "10-Q") else self.max_chunks_8k

        # No filtering needed if already small
        if n_before <= max_cap:
            return {
                "chunks": chunks,
                "n_before": n_before,
                "n_after": n_before,
                "reduction_pct": 0.0,
                "kept_indices": [c["chunk_id"] for c in chunks],
            }

        texts = [c["text"] for c in chunks]

        # ── Step 1: TF-IDF vectorise ──────────────────────────────────
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            vectorizer   = TfidfVectorizer(max_features=512, stop_words="english",
                                           sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(texts)   # (n_chunks, vocab)
        except Exception as e:
            # sklearn not available — fall back to uniform sampling
            print(f"[ChunkFilter] TF-IDF unavailable ({e}), using uniform sampling")
            kept = self._uniform_sample(chunks, max_cap)
            return self._build_result(chunks, kept, n_before)

        # ── Step 2: Score by information density ─────────────────────
        scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

        # ── Step 3: Greedy deduplication ─────────────────────────────
        sorted_idx = np.argsort(scores)[::-1]   # highest-info first
        kept_idx   = []
        kept_vecs  = []

        for idx in sorted_idx:
            if len(kept_idx) >= max_cap:
                break
            vec = tfidf_matrix[idx]
            if not kept_vecs:
                kept_idx.append(idx)
                kept_vecs.append(vec)
            else:
                from scipy.sparse import vstack
                stacked  = vstack(kept_vecs)
                max_sim  = float(cos_sim(vec, stacked).max())
                if max_sim < self.sim_threshold:
                    kept_idx.append(idx)
                    kept_vecs.append(vec)

        # ── Step 4: Restore document order ───────────────────────────
        kept_idx.sort()
        kept_chunks = [chunks[i] for i in kept_idx]
        return self._build_result(chunks, kept_chunks, n_before)

    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_sample(chunks: list, n: int) -> list:
        """Fallback: take every k-th chunk to reach target n."""
        step = max(1, len(chunks) // n)
        return chunks[::step][:n]

    @staticmethod
    def _build_result(original: list, kept: list, n_before: int) -> dict:
        reduction = (n_before - len(kept)) / n_before * 100
        return {
            "chunks": kept,
            "n_before": n_before,
            "n_after": len(kept),
            "reduction_pct": round(reduction, 1),
            "kept_indices": [c["chunk_id"] for c in kept],
        }
