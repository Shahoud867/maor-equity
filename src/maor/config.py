"""Typed, validated configuration.

Replaces the previous mix of module-level constants, hard-coded paths and magic
numbers scattered across agents and scripts. Every knob that can change a
measured number lives here, so the config hash recorded in a result's provenance
fully identifies the setting that produced it.

Configs are dataclasses with explicit validation, loaded from YAML. Unknown keys
are an error, not a silent no-op — a typo in an experiment config should fail the
run rather than quietly measure the default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration is invalid or internally inconsistent."""


@dataclass
class ModelConfig:
    """Model checkpoints and how they are loaded.

    ``estimated_vram_mb`` is what the VRAM budget reserves before loading. It is a
    declared expectation, checked against reality by ``vram-verify``; a large
    divergence is a bug worth surfacing, not a number to quietly adjust.
    """

    sentiment_market: str = "ProsusAI/finbert"
    sentiment_regulatory: str = "yiyanghkust/finbert-tone"
    # Forward-looking-statement classifier. Genuinely distinct from the market
    # checkpoint (finding N3: it previously reused market's weights, giving
    # 985/985 identical labels on PhraseBank — see docs/AUDIT_RESPONSE.md).
    sentiment_temporal: str = "yiyanghkust/finbert-fls"
    summarizer: str = "microsoft/Phi-3-mini-4k-instruct"

    # Quantisation. "none" | "int8" | "nf4"
    sentiment_quantisation: str = "none"
    summarizer_quantisation: str = "nf4"

    # Three BERT-base checkpoints resident during Phase A (market + regulatory +
    # temporal), ~220-260 MB each at FP16-equivalent. Corrected from 550 MB (two
    # checkpoints) now that temporal is a real third model rather than a reused
    # pointer; vram-verify measures the true figure.
    sentiment_estimated_vram_mb: float = 800.0
    summarizer_estimated_vram_mb: float = 2800.0

    max_input_tokens: int = 3500
    sentiment_max_length: int = 512
    sentiment_batch_size: int = 8

    map_max_new_tokens: int = 200
    reduce_max_new_tokens: int = 400
    guardrail_max_new_tokens: int = 200

    # Deterministic decoding: required for the reproducibility claim.
    do_sample: bool = False
    temperature: float = 0.0

    trust_remote_code: bool = True

    _VALID_QUANT = ("none", "int8", "nf4")

    def validate(self) -> None:
        for name, value in (
            ("sentiment_quantisation", self.sentiment_quantisation),
            ("summarizer_quantisation", self.summarizer_quantisation),
        ):
            if value not in self._VALID_QUANT:
                raise ConfigError(
                    f"{name}={value!r} is not one of {self._VALID_QUANT}"
                )
        if self.do_sample and self.temperature <= 0:
            raise ConfigError("do_sample=True requires temperature > 0")
        if self.sentiment_batch_size < 1:
            raise ConfigError("sentiment_batch_size must be >= 1")


@dataclass
class VRAMConfig:
    """GPU memory policy. Nothing here is hard-coded to a specific card.

    ``total_mb=None`` means "probe the device". Set it explicitly only to simulate
    a smaller card than is physically present (useful for testing the budget
    logic itself).
    """

    total_mb: float | None = None
    usable_fraction: float = 0.85
    device: int = 0
    enforce: bool = True

    # Phase serialisation: release each model's memory before the next loads.
    # This is the mechanism that lets a 4 GB card run both models.
    phase_serialised: bool = True

    def validate(self) -> None:
        if not 0.0 < self.usable_fraction <= 1.0:
            raise ConfigError(
                f"usable_fraction must be in (0, 1], got {self.usable_fraction}"
            )
        if self.total_mb is not None and self.total_mb <= 0:
            raise ConfigError("total_mb must be positive or None (probe)")
        if self.device < 0:
            raise ConfigError("device index must be >= 0")


@dataclass
class ChunkingConfig:
    window_tokens: int = 512
    stride_tokens: int = 64
    sim_threshold: float = 0.85
    max_chunks_8k: int = 12
    max_chunks_10k: int = 20
    tfidf_max_features: int = 512

    def validate(self) -> None:
        if self.stride_tokens >= self.window_tokens:
            raise ConfigError(
                f"stride_tokens ({self.stride_tokens}) must be < window_tokens "
                f"({self.window_tokens}), otherwise chunks do not overlap or skip text"
            )
        if not 0.0 < self.sim_threshold <= 1.0:
            raise ConfigError("sim_threshold must be in (0, 1]")
        if self.max_chunks_8k < 1 or self.max_chunks_10k < 1:
            raise ConfigError("chunk caps must be >= 1")


@dataclass
class ExecutionConfig:
    """How and where the pipeline runs."""

    # "local" runs everything in-process (no Ray). "ray" uses a Ray cluster.
    mode: str = "local"
    ray_address: str | None = None
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # No stage may run unbounded. A hung actor should fail the run, not stall it.
    stage_timeout_s: float = 1800.0
    model_load_timeout_s: float = 900.0

    seed: int = 20260428
    n_repeats: int = 1

    _VALID_MODES = ("local", "ray")
    _VALID_DEVICES = ("auto", "cuda", "cpu")

    def validate(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ConfigError(f"mode={self.mode!r} not in {self._VALID_MODES}")
        if self.device not in self._VALID_DEVICES:
            raise ConfigError(f"device={self.device!r} not in {self._VALID_DEVICES}")
        if self.stage_timeout_s <= 0 or self.model_load_timeout_s <= 0:
            raise ConfigError("timeouts must be positive")
        if self.n_repeats < 1:
            raise ConfigError("n_repeats must be >= 1")


@dataclass
class DataConfig:
    ectsum_path: Path = PROJECT_ROOT / "data" / "ectsum" / "ectsum_test.jsonl"
    filings_dir: Path = PROJECT_ROOT / "data" / "sec-edgar-filings"
    phrasebank_config: str = "sentences_75agree"
    sec_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "MAOR_SEC_USER_AGENT", "maor-equity research (contact via repository)"
        )
    )

    def validate(self) -> None:
        # Paths are not required to exist at config time; commands that need them
        # check and give an actionable message. Validate shape only.
        if not str(self.phrasebank_config).startswith("sentences_"):
            raise ConfigError(
                f"phrasebank_config={self.phrasebank_config!r} should be one of "
                f"sentences_50agree / sentences_66agree / sentences_75agree / allagree"
            )


@dataclass
class PathsConfig:
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "paper" / "figures"
    tables_dir: Path = PROJECT_ROOT / "paper" / "tables"
    logs_dir: Path = PROJECT_ROOT / "logs"

    def validate(self) -> None:
        return None

    def ensure(self) -> None:
        for p in (self.results_dir, self.figures_dir, self.tables_dir, self.logs_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Root configuration."""

    models: ModelConfig = field(default_factory=ModelConfig)
    vram: VRAMConfig = field(default_factory=VRAMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def validate(self) -> "Config":
        for f in fields(self):
            section = getattr(self, f.name)
            if hasattr(section, "validate"):
                section.validate()
        self._validate_cross_section()
        return self

    def _validate_cross_section(self) -> None:
        """Checks that span sections — where the real mistakes live."""
        if self.vram.enforce and self.vram.total_mb is not None:
            usable = self.vram.total_mb * self.vram.usable_fraction
            both = (
                self.models.sentiment_estimated_vram_mb
                + self.models.summarizer_estimated_vram_mb
            )
            if not self.vram.phase_serialised and both > usable:
                raise ConfigError(
                    f"phase_serialised=False requires both models resident "
                    f"({both:.0f} MB) but the budget is {usable:.0f} MB. "
                    f"Either enable phase serialisation or use a larger GPU. "
                    f"This is the exact condition that caused the original stalls."
                )
            largest = max(
                self.models.sentiment_estimated_vram_mb,
                self.models.summarizer_estimated_vram_mb,
            )
            if largest > usable:
                raise ConfigError(
                    f"The largest single model needs {largest:.0f} MB but the budget "
                    f"is {usable:.0f} MB. Phase serialisation cannot help: no ordering "
                    f"of loads fits. Use stronger quantisation or a larger GPU."
                )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Load YAML config, apply dotted overrides, validate.

        Overrides use dotted keys: ``Config.load(path, **{"vram.usable_fraction": 0.7})``
        """
        data: dict[str, Any] = {}
        if path is not None:
            data = _read_yaml(Path(path))
        elif DEFAULT_CONFIG_PATH.exists():
            data = _read_yaml(DEFAULT_CONFIG_PATH)

        cfg = _build(cls, data)
        for dotted, value in overrides.items():
            cfg.set(dotted, value)
        return cfg.validate()

    def set(self, dotted: str, value: Any) -> None:
        """Set ``a.b.c`` to ``value``, rejecting unknown paths."""
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ConfigError(f"unknown config section {part!r} in {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ConfigError(f"unknown config key {dotted!r}")
        setattr(target, leaf, value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("pyyaml is required to load YAML configs") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, erroring on unknown keys.

    Annotations are resolved with ``get_type_hints`` rather than read off
    ``field.type``: this module uses ``from __future__ import annotations``, so
    ``field.type`` is the *string* "PathsConfig", and any heuristic over that
    string misclassifies section types.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)  # type: ignore[arg-type]
        elif _is_path_type(ftype) and value is not None:
            kwargs[name] = Path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _is_path_type(ftype: Any) -> bool:
    """True for ``Path`` and ``Path | None``, false for everything else."""
    if ftype is Path:
        return True
    if get_origin(ftype) is not None:
        return Path in get_args(ftype)
    return False


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        # Relative to the project root so the config hash is machine-independent.
        try:
            return str(obj.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return obj.name
    return obj
