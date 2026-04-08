"""
summarization_agent.py  —  Node B (GPU)
Chunked map-reduce summarization with Phi-3-mini-4k-instruct (4-bit).
"""
import ray
import torch


@ray.remote(num_gpus=0.5)
class SummarizationAgent:

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.mdl = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", load_in_4bit=True, device_map="auto"
        )
        print("[SummarizationAgent] Phi-3-mini ready")

    # ------------------------------------------------------------------
    def _gen(self, prompt: str, max_new: int = 300) -> str:
        inp = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=3500).to("cuda")
        with torch.no_grad():
            out = self.mdl.generate(**inp, max_new_tokens=max_new,
                                    temperature=0.1, do_sample=False)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    # ------------------------------------------------------------------
    def map_chunk(self, chunk: dict) -> dict:
        prompt = (
            "You are a financial analyst. Summarize this earnings call segment.\n"
            "Extract: (1) key financial metrics with values, "
            "(2) directional claims (growth/decline/flat), (3) named entities.\n"
            "If contradictory claims exist, preserve BOTH with [CONFLICT] tag.\n"
            "Output: 3-5 bullet points maximum.\n\n"
            f"Segment: {chunk['text']}\n\nSummary:"
        )
        s = self._gen(prompt, 200)
        return {"chunk_id": chunk["chunk_id"], "summary": s,
                "has_conflict": "[CONFLICT]" in s}

    # ------------------------------------------------------------------
    def reduce(self, summaries: list, depth: int = 3) -> dict:
        combined = "\n".join(f"Chunk {s['chunk_id']}: {s['summary']}"
                             for s in summaries)
        if len(self.tok.encode(combined)) > 3000 and depth > 0:
            mid   = len(summaries) // 2
            left  = self.reduce(summaries[:mid],  depth - 1)["summary"]
            right = self.reduce(summaries[mid:], depth - 1)["summary"]
            summaries = [{"chunk_id": "L", "summary": left,  "has_conflict": False},
                         {"chunk_id": "R", "summary": right, "has_conflict": False}]
            combined = "\n".join(f"Section {s['chunk_id']}: {s['summary']}"
                                 for s in summaries)
        n_cf = sum(1 for s in summaries if s.get("has_conflict"))
        prompt = (
            "Synthesize earnings call chunk summaries into a final equity research summary.\n"
            "IMPORTANT: Keep all [CONFLICT] tags — do not resolve contradictions.\n\n"
            f"Summaries:\n{combined}\n\n"
            "Produce:\n"
            "1. Executive summary (2 sentences)\n"
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
        refs = {c["chunk_id"]: ray.put(self.map_chunk(c)) for c in chunks}
        out  = self.reduce([ray.get(r) for r in refs.values()])
        out["doc_id"] = doc_id
        return out