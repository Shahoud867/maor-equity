"""
guardrail_agent.py  —  Node B (GPU)
Bull / Bear assessment with weighted rule-based arbiter using Phi-3-mini.
"""
import json
import ray
import numpy as np


@ray.remote(num_gpus=0.2)
class GuardrailAgent:

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.mdl = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", load_in_4bit=True, device_map="auto"
        )

    def _gen_json(self, prompt: str) -> dict:
        import torch
        inp = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=2000).to("cuda")
        with torch.no_grad():
            out = self.mdl.generate(**inp, max_new_tokens=200, temperature=0.1)
        txt = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(txt[txt.find("{") : txt.rfind("}") + 1])
        except Exception:
            return {"direction": "unknown", "confidence": 0.0, "signals": []}

    def assess(self, summary: str, sentiment_vector, tech: dict) -> dict:
        sv = sentiment_vector
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
        if diff > 0.25:
            w = "bullish" if bs > brs else "bearish"
            return {"recommendation": w,
                    "confidence": "HIGH" if diff > 0.40 else "MEDIUM",
                    "bull_score": round(bs, 3), "bear_score": round(brs, 3),
                    "winning_signals": (bull if bs > brs else bear).get("signals", []),
                    "rsi": rsi, "conflict": False}
        return {"recommendation": "UNRESOLVED", "confidence": "LOW",
                "bull_score": round(bs, 3), "bear_score": round(brs, 3),
                "conflict": True,
                "conflict_signals": bull.get("signals", []) + bear.get("signals", []),
                "rsi": rsi, "note": "Manual review recommended"}