"""Tests for the CLI contract.

These guard the interface the GPU runbook tells a user to type. A command that
works only when the flags are in one particular order is a command that fails at
the moment someone is following instructions on an unfamiliar machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maor.cli import _coerce, _load_config, build_parser


class TestArgumentOrder:
    """Global flags must work before and after the subcommand.

    Every example in docs/GPU_RUNBOOK.md is written as
    ``<command> --config configs/gpu_t1000.yaml``, which argparse rejects by
    default because global options belong to the top-level parser.
    """

    @pytest.mark.parametrize(
        "argv",
        [
            ["gpu-audit", "--config", "configs/gpu_t1000.yaml"],
            ["--config", "configs/gpu_t1000.yaml", "gpu-audit"],
        ],
    )
    def test_config_flag_accepted_in_both_positions(self, argv):
        args = build_parser().parse_args(argv)
        assert getattr(args, "config", None) == Path("configs/gpu_t1000.yaml")

    def test_config_before_subcommand_is_not_clobbered(self):
        """A parent parser's defaults are re-applied by the subparser.

        With a plain ``default=None`` the subparser overwrites the value given
        before the subcommand, silently falling back to the default config.
        """
        args = build_parser().parse_args(
            ["--config", "configs/gpu_t1000.yaml", "doctor"]
        )
        cfg = _load_config(args)
        assert cfg.vram.usable_fraction == 0.78, "T1000 config must survive"

    def test_config_after_subcommand_resolves_identically(self):
        args = build_parser().parse_args(
            ["doctor", "--config", "configs/gpu_t1000.yaml"]
        )
        cfg = _load_config(args)
        assert cfg.vram.usable_fraction == 0.78

    def test_absent_config_falls_back_to_default(self):
        args = build_parser().parse_args(["doctor"])
        cfg = _load_config(args)
        assert cfg.vram.usable_fraction == 0.85

    @pytest.mark.parametrize(
        "argv",
        [
            ["h3-sentiment", "--set", "execution.seed=7"],
            ["--set", "execution.seed=7", "h3-sentiment"],
        ],
    )
    def test_set_override_accepted_in_both_positions(self, argv):
        cfg = _load_config(build_parser().parse_args(argv))
        assert cfg.execution.seed == 7

    def test_verbose_flag_is_optional_everywhere(self):
        for argv in (["doctor"], ["doctor", "-v"], ["-v", "doctor"], ["-vv", "doctor"]):
            args = build_parser().parse_args(argv)
            assert isinstance(getattr(args, "verbose", 0), int)


class TestSubcommands:
    def test_every_documented_command_exists(self):
        """The runbook and README name these; a missing one is a broken doc."""
        expected = {
            "doctor",
            "h3-sentiment",
            "chunk-filter",
            "vram-verify",
            "fetch-models",
            "smoke",
            "h2-summarisation",
            "h1-latency",
            "verify-cluster",
            "report",
            "gpu-audit",
            "run-all",
        }
        parser = build_parser()
        actions = [
            a for a in parser._actions if isinstance(a, type(parser._subparsers._group_actions[0]))  # type: ignore[union-attr]
        ]
        available = set(actions[0].choices)
        assert expected <= available, f"missing: {expected - available}"

    def test_every_subcommand_has_a_handler(self):
        parser = build_parser()
        sub = parser._subparsers._group_actions[0]  # type: ignore[union-attr]
        for name, subparser in sub.choices.items():
            defaults = subparser._defaults
            assert "func" in defaults, f"{name} has no handler"

    def test_run_all_accepts_experiment_selection(self):
        args = build_parser().parse_args(
            ["run-all", "--only", "chunk_filter", "h3_sentiment"]
        )
        assert args.only == ["chunk_filter", "h3_sentiment"]

    def test_run_all_resume_is_default(self):
        args = build_parser().parse_args(["run-all"])
        assert args.no_resume is False


class TestValueCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("false", False),
            ("none", None),
            ("42", 42),
            ("0.78", 0.78),
            ("cuda", "cuda"),
            ("ProsusAI/finbert", "ProsusAI/finbert"),
        ],
    )
    def test_override_values_are_coerced(self, raw, expected):
        assert _coerce(raw) == expected

    def test_unknown_key_is_rejected(self):
        from maor.config import ConfigError

        args = build_parser().parse_args(["doctor", "--set", "vram.not_a_key=1"])
        with pytest.raises(ConfigError, match="unknown config key"):
            _load_config(args)

    def test_malformed_override_is_rejected(self):
        from maor.config import ConfigError

        args = build_parser().parse_args(["doctor", "--set", "novalue"])
        with pytest.raises(ConfigError, match="key=value"):
            _load_config(args)
