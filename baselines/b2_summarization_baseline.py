"""B2: Single-pass Phi-3-mini zero-shot summarisation — ROUGE baseline."""
import json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_single_pass(transcript: str) -> str:
    """Returns just the summary string — used by rouge_eval.py."""
    result = run(transcript)
    return result["summary"]


def run(transcript: str) -> dict:
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    mdl = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", load_in_4bit=True, device_map="auto"
    )
    tokens = tok.encode(transcript, max_length=3500, truncation=True)
    text   = tok.decode(tokens)
    prompt = ("Summarize the following earnings call transcript as bullet points "
              "covering key financial metrics, directional outlook, and risk factors.\n\n"
              f"Transcript: {text}\n\nSummary:")
    t0  = time.time()
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=300, temperature=0.1)
    summary = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"method": "B2_single_pass", "summary": summary,
            "input_tokens": len(tokens), "truncated": len(tokens) >= 3500,
            "latency_s": time.time() - t0}


if __name__ == "__main__":
    with open("data/ectsum/ectsum_test.jsonl") as f:
        sample = json.loads(f.readline())
    r = run(sample["transcript"])
    print(r["summary"])
    print(f"\n>>> Record B2 latency: {r['latency_s']:.2f}s   truncated={r['truncated']}")