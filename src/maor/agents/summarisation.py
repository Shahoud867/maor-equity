"""Map-reduce summarisation over a shared causal LM.

The model is wrapped in :class:`SummarisationModel`, which both the summariser and
the guardrail hold a reference to. Loading it twice would double a ~2.7 GB
residency and is the single easiest way to exceed a 4 GB budget, so the VRAM
registry treats a second load as an error rather than an out-of-memory crash.

Audit finding M6: the AAPL summary in ``results/aapl.json`` ran off into leaked
instruction-tuning scaffolding ("**Instruction 2 (More Difficult):**...") because
generation continued past the end of the answer and nothing checked the output.
:func:`clean_generation` strips known scaffolding and
:meth:`SummarisationModel.generate` records whether it fired, so contamination is
visible in the results rather than published as a summary.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

log = logging.getLogger(__name__)

# Patterns marking the point where an instruction-tuned model stops answering and
# starts inventing new tasks. Everything from the first match onward is dropped.
_SCAFFOLDING_PATTERNS = (
    r"\*\*Instruction\s*\d*",
    r"\*\*Response\s*\d*\s*:",
    r"\n#{1,6}\s*Instruction",
    r"<\|(?:user|assistant|system|end|endoftext)\|>",
    r"\nInstruction\s*\d+\s*[:(]",
    r"\nTask\s*\d+\s*:",
    r"\n\s*(?:Follow[- ]up|Additional)\s+(?:Instruction|Question)s?\s*\d*\s*:",
)
_SCAFFOLDING_RE = re.compile("|".join(_SCAFFOLDING_PATTERNS), re.IGNORECASE)


@dataclass
class GenerationResult:
    text: str
    raw_text: str
    scaffolding_removed: bool
    n_input_tokens: int
    n_generated_tokens: int
    truncated_input: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaffolding_removed": self.scaffolding_removed,
            "n_input_tokens": self.n_input_tokens,
            "n_generated_tokens": self.n_generated_tokens,
            "truncated_input": self.truncated_input,
        }


def clean_generation(text: str) -> tuple[str, bool]:
    """Trim instruction-tuning scaffolding. Returns (cleaned, was_modified)."""
    match = _SCAFFOLDING_RE.search(text)
    if match is None:
        return text.strip(), False
    return text[: match.start()].strip(), True


class SummarisationModel:
    """A single resident causal LM, shared by every agent that needs generation."""

    def __init__(
        self,
        checkpoint: str = "microsoft/Phi-3-mini-4k-instruct",
        *,
        device: str = "cuda",
        quantisation: str = "nf4",
        max_input_tokens: int = 3500,
        trust_remote_code: bool = True,
        do_sample: bool = False,
        temperature: float = 0.0,
        estimated_vram_mb: float = 2800.0,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.quantisation = quantisation
        self.max_input_tokens = max_input_tokens
        self.trust_remote_code = trust_remote_code
        self.do_sample = do_sample
        self.temperature = temperature
        self.estimated_vram_mb = estimated_vram_mb
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "SummarisationModel":
        if self.is_loaded:
            log.debug("%s already loaded; reusing", self.checkpoint)
            return self

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=self.trust_remote_code
        )

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.device == "cpu":
            if self.quantisation != "none":
                raise ValueError(
                    f"quantisation={self.quantisation!r} requires CUDA. On CPU use "
                    f"--set models.summarizer_quantisation=none (expect it to be slow)."
                )
            kwargs["dtype"] = torch.float32
        else:
            if self.quantisation != "none":
                from transformers import BitsAndBytesConfig

                if self.quantisation == "nf4":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif self.quantisation == "int8":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            # Pin every layer to one device. device_map="auto" spilled layers to
            # the integrated GPU on the original Node B, adding a PCIe copy to
            # every forward pass.
            kwargs["device_map"] = {"": 0}

        self._model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)
        self._model.eval()

        # Record residency so a second load of the same checkpoint is refused
        # rather than silently doubling a ~2.7 GB footprint. allow_shared is set
        # because the guardrail deliberately uses this same instance.
        if self.device != "cpu":
            from ..gpu.lifecycle import ModelRegistry

            device_index = int(str(self.device).split(":")[-1] or 0)
            ModelRegistry.instance().register(
                label="summariser",
                checkpoint=self.checkpoint,
                device=device_index,
                estimated_mb=self.estimated_vram_mb,
                obj=self._model,
                allow_shared=True,
            )

        log.info(
            "loaded %s (device=%s, quantisation=%s)",
            self.checkpoint,
            self.device,
            self.quantisation,
        )
        return self

    def generate(self, prompt: str, *, max_new_tokens: int = 200) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("model not loaded; call load() first")

        import torch

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        full_length = len(self._tokenizer(prompt)["input_ids"])
        truncated = full_length > self.max_input_tokens

        if self.device != "cpu":
            encoded = {k: v.to(self._model.device) for k, v in encoded.items()}

        n_input = int(encoded["input_ids"].shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            out = self._model.generate(**encoded, **gen_kwargs)

        generated_ids = out[0][n_input:]
        raw = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned, modified = clean_generation(raw)
        if modified:
            log.warning(
                "generation contained instruction scaffolding; trimmed %d -> %d chars",
                len(raw),
                len(cleaned),
            )

        return GenerationResult(
            text=cleaned,
            raw_text=raw,
            scaffolding_removed=modified,
            n_input_tokens=n_input,
            n_generated_tokens=int(generated_ids.shape[0]),
            truncated_input=truncated,
        )

    def unload(self, *, strict: bool = False) -> Any:
        """Release the model from the device and verify the memory came back.

        Clearing the attributes is not sufficient on its own: the model is moved
        to CPU first so the device memory is freed while the reference is still
        valid, rather than depending on the collector to reach it later. Returns
        the verification so a caller can record whether the release worked.
        """
        from ..gpu.lifecycle import ModelRegistry, release_torch_module

        model, tokenizer = self._model, self._tokenizer
        self._model = None
        self._tokenizer = None

        if model is None and tokenizer is None:
            return None

        device_index = 0 if self.device == "cpu" else int(str(self.device).split(":")[-1] or 0)
        ModelRegistry.instance().unregister(self.checkpoint, device_index)

        verification = release_torch_module(
            f"summariser[{self.checkpoint}]",
            model,
            tokenizer,
            device=device_index,
            strict=strict,
        )
        log.info(
            "unloaded %s: freed %.0f MB allocated, %.0f MB still held",
            self.checkpoint,
            verification.allocated_freed_mb,
            verification.residual_mb,
        )
        return verification

    def __enter__(self) -> "SummarisationModel":
        return self.load()

    def __exit__(self, *exc: object) -> bool:
        self.unload()
        return False


MAP_PROMPT = (
    "<|user|>\nYou are a financial analyst. Summarise this segment of a filing in "
    "at most four sentences. State key metrics, directional claims and named "
    "entities. If the segment contradicts itself, prefix the summary with "
    "[CONFLICT].\n\nSegment:\n{chunk}\n<|end|>\n<|assistant|>\n"
)

REDUCE_PROMPT = (
    "<|user|>\nYou are a financial analyst. Combine these segment summaries into "
    "one executive summary. Include: key financial metrics, directional outlook, "
    "and risk flags. Do not invent figures that do not appear below.\n\n"
    "Segment summaries:\n{summaries}\n<|end|>\n<|assistant|>\n"
)


@dataclass
class SummaryResult:
    summary: str
    n_chunks: int
    n_conflicts: int
    n_scaffolding_trimmed: int
    reduce_depth: int
    chunk_summaries: list[str] = field(default_factory=list)
    generation_stats: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "n_chunks": self.n_chunks,
            "n_conflicts": self.n_conflicts,
            "n_scaffolding_trimmed": self.n_scaffolding_trimmed,
            "reduce_depth": self.reduce_depth,
        }


class MapReduceSummariser:
    """Map each chunk to a summary, then reduce the summaries into one.

    Recursive reduction keeps the reduce prompt inside the context window; the
    depth limit prevents unbounded recursion on pathological inputs.
    """

    def __init__(
        self,
        model: SummarisationModel,
        *,
        map_max_new_tokens: int = 200,
        reduce_max_new_tokens: int = 400,
        max_reduce_depth: int = 3,
        reduce_char_budget: int = 12_000,
    ) -> None:
        self.model = model
        self.map_max_new_tokens = map_max_new_tokens
        self.reduce_max_new_tokens = reduce_max_new_tokens
        self.max_reduce_depth = max_reduce_depth
        self.reduce_char_budget = reduce_char_budget

    def summarise(self, chunk_texts: Sequence[str]) -> SummaryResult:
        if not chunk_texts:
            raise ValueError("cannot summarise zero chunks")

        stats: list[dict[str, Any]] = []
        trimmed = 0
        summaries: list[str] = []

        for i, chunk in enumerate(chunk_texts):
            res = self.model.generate(
                MAP_PROMPT.format(chunk=chunk), max_new_tokens=self.map_max_new_tokens
            )
            summaries.append(res.text)
            stats.append({"phase": "map", "chunk": i, **res.to_dict()})
            trimmed += int(res.scaffolding_removed)

        n_conflicts = sum(1 for s in summaries if "[CONFLICT]" in s.upper())
        final, depth, reduce_stats, reduce_trimmed = self._reduce(summaries, depth=0)
        stats.extend(reduce_stats)

        return SummaryResult(
            summary=final,
            n_chunks=len(chunk_texts),
            n_conflicts=n_conflicts,
            n_scaffolding_trimmed=trimmed + reduce_trimmed,
            reduce_depth=depth,
            chunk_summaries=summaries,
            generation_stats=stats,
        )

    def _reduce(
        self, summaries: Sequence[str], *, depth: int
    ) -> tuple[str, int, list[dict[str, Any]], int]:
        joined = "\n\n".join(f"- {s}" for s in summaries)
        stats: list[dict[str, Any]] = []
        trimmed = 0

        if len(joined) > self.reduce_char_budget and depth < self.max_reduce_depth:
            mid = len(summaries) // 2
            if mid == 0:
                mid = 1
            left, d1, s1, t1 = self._reduce(summaries[:mid], depth=depth + 1)
            right, d2, s2, t2 = self._reduce(summaries[mid:], depth=depth + 1)
            stats.extend(s1 + s2)
            trimmed += t1 + t2
            return self._reduce_once([left, right], depth=max(d1, d2), stats=stats, trimmed=trimmed)

        return self._reduce_once(summaries, depth=depth, stats=stats, trimmed=trimmed)

    def _reduce_once(
        self,
        summaries: Sequence[str],
        *,
        depth: int,
        stats: list[dict[str, Any]],
        trimmed: int,
    ) -> tuple[str, int, list[dict[str, Any]], int]:
        joined = "\n\n".join(f"- {s}" for s in summaries)
        res = self.model.generate(
            REDUCE_PROMPT.format(summaries=joined),
            max_new_tokens=self.reduce_max_new_tokens,
        )
        stats.append({"phase": "reduce", "depth": depth, **res.to_dict()})
        return res.text, depth, stats, trimmed + int(res.scaffolding_removed)


def single_pass_summarise(
    model: SummarisationModel, document: str, *, max_new_tokens: int = 400
) -> SummaryResult:
    """B2 baseline: one call on the truncated document, no chunking.

    Truncation happens inside :meth:`SummarisationModel.generate` at
    ``max_input_tokens``, and ``truncated_input`` records whether it fired — the
    quantity that makes the map-reduce comparison meaningful.
    """
    res = model.generate(
        REDUCE_PROMPT.format(summaries=document), max_new_tokens=max_new_tokens
    )
    return SummaryResult(
        summary=res.text,
        n_chunks=1,
        n_conflicts=0,
        n_scaffolding_trimmed=int(res.scaffolding_removed),
        reduce_depth=0,
        generation_stats=[{"phase": "single_pass", **res.to_dict()}],
    )
