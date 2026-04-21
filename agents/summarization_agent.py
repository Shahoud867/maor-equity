"""
summarization_agent.py  —  Node B (GPU)
Chunked map-reduce summarization with Phi-3-mini-4k-instruct (4-bit).
"""
import ray


@ray.remote(num_gpus=0.02)  # tiny fraction forces Node B placement (near shared model)
class SummarizationAgent:

    def __init__(self, phi3_actor):
        self.phi3 = phi3_actor
        print("[SummarizationAgent] ready (using shared Phi3ModelActor)")

    def ping(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def _gen(self, prompt: str, max_new: int = 300) -> str:
        return ray.get(self.phi3.generate.remote(prompt, max_new))

    # ------------------------------------------------------------------
    def map_chunk(self, chunk: dict, doc_id: str = "") -> dict:
        # Truncate chunk text to ~800 tokens (3200 chars) so prompt + text
        # stays within Phi-3-mini's 4096-token context window.
        text = chunk["text"][:3200]
        prompt = (
            f"You are a financial analyst reviewing an SEC filing ({doc_id}).\n"
            "Summarize the following filing segment.\n"
            "Extract: (1) key financial metrics with values, "
            "(2) directional claims (growth/decline/flat), (3) named entities.\n"
            "If contradictory claims exist, preserve BOTH with [CONFLICT] tag.\n"
            "Output: 3-5 bullet points maximum. Use only information in the text below.\n\n"
            f"Filing segment:\n{text}\n\nSummary:"
        )
        s = self._gen(prompt, 200)
        return {"chunk_id": chunk["chunk_id"], "summary": s,
                "has_conflict": "[CONFLICT]" in s}

    # ------------------------------------------------------------------
    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Fast character-based token estimate (≈ 1 token per 4 chars).
        Replaces self.tok.encode() — tok was removed when we switched to
        shared Phi3ModelActor; this avoids loading a tokenizer on Node B
        just for a length check."""
        return len(text) // 4

    def reduce(self, summaries: list, depth: int = 3) -> dict:
        combined = "\n".join(f"Chunk {s['chunk_id']}: {s['summary']}"
                             for s in summaries)
        if self._approx_tokens(combined) > 3000 and depth > 0:
            mid   = len(summaries) // 2
            left  = self.reduce(summaries[:mid],  depth - 1)["summary"]
            right = self.reduce(summaries[mid:], depth - 1)["summary"]
            summaries = [{"chunk_id": "L", "summary": left,  "has_conflict": False},
                         {"chunk_id": "R", "summary": right, "has_conflict": False}]
            combined = "\n".join(f"Section {s['chunk_id']}: {s['summary']}"
                                 for s in summaries)
        n_cf = sum(1 for s in summaries if s.get("has_conflict"))
        prompt = (
            "Synthesize the following SEC filing chunk summaries into a final equity research note.\n"
            "IMPORTANT: Keep all [CONFLICT] tags. Do not invent facts not present in the summaries.\n\n"
            f"Chunk summaries:\n{combined}\n\n"
            "Produce:\n"
            "1. Executive summary (2 sentences, based only on the summaries above)\n"
            "2. Key financial metrics (bullets)\n"
            "3. Directional outlook (bullish/bearish/mixed)\n"
            "4. Risk flags (list any [CONFLICT] items)\n\n"
            "Final Summary:"
        )
        return {"summary": self._gen(prompt, 400),
                "n_chunks": len(summaries), "n_conflicts": n_cf}

    # ------------------------------------------------------------------
    def process_document(self, chunks: list, doc_id: str) -> dict:
        import ray
        refs = {c["chunk_id"]: ray.put(self.map_chunk(c, doc_id)) for c in chunks}
        out  = self.reduce([ray.get(r) for r in refs.values()])
        out["doc_id"] = doc_id
        return out