"""
guardrail_agent.py  —  Node B (GPU)
Bull / Bear assessment with weighted rule-based arbiter using Phi-3-mini.

VRAM note: GuardrailAgent shares the Phi-3-mini model with SummarizationAgent
via a dedicated Phi3ModelActor to stay within the 4 GB T1000 budget.
Both agents receive a handle to the same actor and call .generate() on it.
"""
import json
import ray
import numpy as np


@ray.remote(num_gpus=0.2, max_restarts=3, max_task_retries=2)
class Phi3ModelActor:
    """Singleton model actor — loaded once, shared by Summarizer and Guardrail."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
        )
        import torch
        from transformers import BitsAndBytesConfig
        # NF4 + double quantization: quantizes the quantization constants too.
        # Saves ~0.37 bits/param extra → ~175 MB less than plain 4-bit on 3.8B model.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            quantization_config=bnb,
            device_map={"": 0},
            trust_remote_code=True,
        )
        print("[Phi3ModelActor] Phi-3-mini loaded once — shared by summarizer + guardrail")

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        import torch
        inp = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=3500).to("cuda")
        with torch.no_grad():
            out = self.mdl.generate(**inp, max_new_tokens=max_new_tokens,
                                    do_sample=False)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


@ray.remote(num_gpus=0.02, max_restarts=3, max_task_retries=2)
class GuardrailAgent:

    def __init__(self, phi3_actor):
        self.phi3 = phi3_actor

    def ping(self) -> bool:
        return True

    def _gen_json(self, prompt: str) -> dict:
        txt = ray.get(self.phi3.generate.remote(prompt, 200))
        try:
            return json.loads(txt[txt.find("{") : txt.rfind("}") + 1])
        except Exception:
            return {"direction": "unknown", "confidence": 0.0, "signals": []}

    def assess(self, summary: str, sentiment_vector, tech: dict) -> dict:
        sv = np.asarray(sentiment_vector, dtype=float)
        ss = (f"Market pos={sv[0,0]:.2f}/neg={sv[0,2]:.2f} | "
              f"Regulatory pos={sv[1,0]:.2f}/neg={sv[1,2]:.2f} | "
              f"Temporal pos={sv[2,0]:.2f}/neg={sv[2,2]:.2f}")
        base = (f"Summary: {summary[:800]}\nSentiment: {ss}\n"
                f"RSI={tech.get('rsi',50):.1f} MACD_bull={tech.get('macd_crossover_bullish',False)}\n")
        bull = self._gen_json(
            "Make the strongest BULL case for this equity.\n" + base +
            'JSON only: {"direction":"bullish","confidence":0.0,"signals":["s1","s2","s3"]}'
        )
        bear = self._gen_json(
            "Make the strongest BEAR case for this equity.\n" + base +
            'JSON only: {"direction":"bearish","confidence":0.0,"signals":["s1","s2","s3"]}'
        )
        return self._arbitrate(bull, bear, tech)

    def _arbitrate(self, bull: dict, bear: dict, tech: dict) -> dict:
        rsi  = tech.get("rsi", 50)
        tw   = abs(rsi - 50) / 50
        td   = 1 if (rsi < 30 or tech.get("macd_crossover_bullish")) else (-1 if rsi > 70 else 0)
        bs   = bull.get("confidence", 0.5) + (0.25 * tw if td > 0 else 0)
        brs  = bear.get("confidence", 0.5) + (0.25 * tw if td < 0 else 0)
        diff = abs(bs - brs)
        if diff > 0.10:
            w = "bullish" if bs > brs else "bearish"
            return {"recommendation": w,
                    "confidence": "HIGH" if diff > 0.30 else ("MEDIUM" if diff > 0.20 else "LOW"),
                    "bull_score": round(bs, 3), "bear_score": round(brs, 3),
                    "winning_signals": (bull if bs > brs else bear).get("signals", []),
                    "rsi": rsi, "conflict": False}
        return {"recommendation": "UNRESOLVED", "confidence": "LOW",
                "bull_score": round(bs, 3), "bear_score": round(brs, 3),
                "conflict": True,
                "conflict_signals": bull.get("signals", []) + bear.get("signals", []),
                "rsi": rsi, "note": "Manual review recommended"}