"""Thin Python CLI wrapper — presentation layer only, zero computation logic.

All computation is delegated to ``thrindex._core`` (PyO3 / Rust).
This module is the entrypoint for the ``thrindex`` and ``thx`` console scripts.

Subcommands
-----------
run      Load a .thx artifact, run the behavioral simulator, print §29 transcript.
doctor   Diagnose the environment (Python version, wheel integrity, torch presence).
targets  List available simulation targets.
"""

from __future__ import annotations

import json
import sys
from typing import NoReturn


def _die(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _parse_first_in_features(artifact_path: str) -> int:
    """Read in_features from the first Dense/Conv2d layer in the artifact JSON.

    Pure Python — no Rust, no torch — so this works even without a full build.
    """
    try:
        data = json.loads(open(artifact_path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"E0008: cannot read artifact {artifact_path!r}: {exc}")
    for layer in data.get("model", {}).get("layers", []):
        t = layer.get("type", "")
        if t == "dense":
            return int(layer["in_features"])
        if t == "conv2d":
            return int(layer["in_channels"])
    return 64  # fallback


def _load_input_file(path: str) -> list[list[list[float]]]:
    """Load spike input from a JSON file or stdin (path == "-").

    Expected format: a 3-D JSON array of shape [batch, T, n_features] where
    every value is 0.0 or 1.0.

    Parameters
    ----------
    path:
        Path to a JSON file, or ``"-"`` to read from stdin.

    Returns
    -------
    list[list[list[float]]]
        Spike raster with shape [batch, T, n_features].

    Raises
    ------
    SystemExit
        On any I/O or parse error, prints an E-code message to stderr and exits.
    """
    try:
        if path == "-":
            raw = sys.stdin.read()
            source = "<stdin>"
        else:
            raw = open(path, encoding="utf-8").read()
            source = repr(path)
    except OSError as exc:
        _die(f"E0001: cannot open input file {path!r}: {exc}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _die(
            f"E0008: input file {source} is not valid JSON: {exc}\n"
            "Expected a 3-D array [[[ float, ... ], ...], ...] of shape [batch, T, n_features]."
        )

    if not isinstance(data, list) or len(data) == 0:
        _die(
            f"E0010: input {source} must be a non-empty JSON array.\n"
            "Expected shape [batch, T, n_features]."
        )

    return data  # type: ignore[return-value]


def _validate_input_shape(
    input_spikes: list[list[list[float]]],
    expected_features: int,
    source: str,
) -> None:
    """Validate the shape of a spike raster before passing it to run_sim.

    Parameters
    ----------
    input_spikes:
        3-D list [batch, T, n_features].
    expected_features:
        Number of input features declared in the artifact.
    source:
        Human-readable label for error messages (file path or "<stdin>").

    Raises
    ------
    SystemExit
        On any shape mismatch, prints an E0010 message to stderr and exits.
    """
    n_samples = len(input_spikes)
    if n_samples == 0:
        _die(f"E0010: input {source!r} is empty (zero samples).")

    first_t = None
    for i, sample in enumerate(input_spikes):
        if not isinstance(sample, list) or len(sample) == 0:
            _die(
                f"E0010: input {source!r} sample[{i}] must be a non-empty list of timesteps.\n"
                "Expected shape [batch, T, n_features]."
            )
        t = len(sample)
        if first_t is None:
            first_t = t
        elif t != first_t:
            _die(
                f"E0010: input {source!r} samples have inconsistent T: "
                f"sample[0] has T={first_t}, sample[{i}] has T={t}.\n"
                "All samples must have the same number of timesteps."
            )
        for j, frame in enumerate(sample):
            if not isinstance(frame, list):
                _die(
                    f"E0010: input {source!r} sample[{i}][{j}] must be a list of floats."
                )
            if len(frame) != expected_features:
                _die(
                    f"E0010: input {source!r} sample[{i}][{j}] has {len(frame)} features, "
                    f"but artifact expects {expected_features}.\n"
                    "Ensure the input was encoded from data with the same feature dimension "
                    "as the model's first layer."
                )


def _gen_demo_input(n_features: int, t_steps: int, seed: int) -> list[list[list[float]]]:
    """LCG spike input — no torch, same algorithm as cmd/run.rs.

    NON-CANONICAL, NON-PRODUCTION encoder.  Used ONLY for the self-contained
    ``thrindex run`` demo path (golden-path golden-path smoke, not real inference).
    Real workloads supply pre-encoded spike trains from ``thrindex.encoders``.

    This is ENCODER logic (inputs are generated before the sim) — it belongs here,
    not in the simulator.  The sim itself has zero RNG (correction 2 / ADR-0007).
    Using the same LCG as the Rust CLI demo ensures the Python and Rust golden paths
    produce the same default input for the same seed.  No test depends on this
    function; any test needing spike data uses ``torch.Generator``-seeded Bernoulli.
    """
    state = (seed + 6_364_136_223_846_793_005) & 0xFFFF_FFFF_FFFF_FFFF
    frames: list[list[float]] = []
    for _ in range(t_steps):
        frame: list[float] = []
        for _ in range(n_features):
            state = (
                state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
            ) & 0xFFFF_FFFF_FFFF_FFFF
            frame.append(1.0 if (state >> 33) % 5 == 0 else 0.0)
        frames.append(frame)
    return [frames]  # batch of 1


def _cmd_run(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="thrindex run",
        description="Run a .thx artifact through the behavioral simulator.",
    )
    parser.add_argument("artifact", help="Path to the .thx artifact.")
    parser.add_argument(
        "--input",
        metavar="FILE",
        default=None,
        help=(
            "Path to a JSON spike input file, or '-' for stdin. "
            "Format: 3-D array [batch, T, n_features] with values 0.0 or 1.0. "
            "When omitted, a deterministic demo input is generated."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Encoder seed (demo mode) or transcript seed.")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Thread count (0 = hardware default).",
    )
    args = parser.parse_args(argv)

    # All computation in Rust (correction 3 / ARCHITECTURE.md layer law).
    try:
        from thrindex._core import run_sim  # type: ignore[import-untyped]
    except ImportError:
        _die("E0001: thrindex._core not found — wheel may be incomplete. Run `thrindex doctor`.")

    n_features = _parse_first_in_features(args.artifact)

    if args.input is not None:
        # User-supplied spike input.
        source = "<stdin>" if args.input == "-" else args.input
        input_spikes = _load_input_file(args.input)
        _validate_input_shape(input_spikes, n_features, source)
    else:
        # Demo path: deterministic LCG input for smoke / quickstart.
        input_spikes = _gen_demo_input(n_features, 100, args.seed)

    _spikes, _stats, transcript = run_sim(
        args.artifact,
        input_spikes,
        args.threads,
        args.seed,
    )
    print(transcript, end="")



def _cmd_doctor(argv: list[str]) -> None:
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        prog="thrindex doctor",
        description="Diagnose the thrindex installation.",
    )
    parser.add_argument(
        "--check",
        metavar="FILE",
        help="Optional .thx artifact to test readability.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detail for every check.")
    args = parser.parse_args(argv)

    border = "═" * 55

    print(border)
    print(f" thrindex doctor v{_sdk_version()}")
    print(border)

    ok_count = 0
    advisory_count = 0
    fail_count = 0

    def _check(label: str, ok: bool, advisory: bool = False, detail: str = "") -> None:
        nonlocal ok_count, advisory_count, fail_count
        if ok:
            icon = "OK "
            ok_count += 1
        elif advisory:
            icon = "?  "
            advisory_count += 1
        else:
            icon = "ERR"
            fail_count += 1
        print(f" [{icon}]  {label}")
        if detail and (args.verbose or not ok):
            for line in detail.splitlines():
                print(f"        {line}")

    # Check 1: Python version >= 3.11.
    py = sys.version_info
    _check(
        f"Python {py.major}.{py.minor}.{py.micro}",
        ok=py >= (3, 11),
        detail="Python 3.11+ required." if py < (3, 11) else "",
    )

    # Check 2: wheel integrity (thrindex._core importable).
    try:
        core = importlib.import_module("thrindex._core")
        _check(
            f"thrindex._core {core.__version__}",
            ok=True,
            detail="Rust extension loaded successfully.",
        )
    except ImportError as exc:
        _check(
            "thrindex._core",
            ok=False,
            detail=(
                f"Import failed: {exc}\n"
                "Re-install with `pip install --force-reinstall thrindex`."
            ),
        )

    # Check 3: torch (needed for authoring/training — optional for `thrindex run`).
    try:
        import torch as _torch  # noqa: PLC0415
        _check(
            f"torch {_torch.__version__}",
            ok=True,
            advisory=False,
            detail="Needed for authoring/training only — not required for `thrindex run`.",
        )
    except ImportError:
        _check(
            "torch: not found",
            ok=False,
            advisory=True,
            detail=(
                "Needed for authoring/training only — not required for `thrindex run`.\n"
                "Install: pip install torch"
            ),
        )

    # Check 4: artifact readability (if --check provided).
    if args.check:
        try:
            from thrindex._core import check_artifact  # type: ignore[import-untyped]

            check_artifact(args.check)
            _check(f"artifact: {args.check}", ok=True, detail="Format OK.")
        except Exception as exc:  # noqa: BLE001
            _check(f"artifact: {args.check}", ok=False, detail=str(exc))

    print(border)
    if fail_count:
        print(f" {fail_count} failure(s). Run `thrindex doctor --verbose` for details.")
    elif advisory_count:
        print(
            f" {advisory_count} advisory. Run `thrindex doctor --verbose` for details."
        )
    else:
        print(" All checks passed.")
    print(border)


def _cmd_targets(argv: list[str]) -> None:
    border = "═" * 55
    print(border)
    print(f" thrindex {_sdk_version()}  —  available targets")
    print(border)
    print(" sim               Behavioral simulator")
    print("                   Precision: float32  |  ADR-0007")
    print("                   Deterministic, CPU-parallel (Rayon)")
    print("─" * 55)
    print(" akida-akd1500     BrainChip AKD1500 neuromorphic chip")
    print("                   Precision: int4  |  RFC-004 / ADR-0011")
    print("                   Requires: --features hardware + Engine Library")
    print("                   Note: LIF layers rejected (non-SNN backend)")
    print(border)


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("thrindex")
    except Exception:  # noqa: BLE001
        return "0+uninstalled"


def main() -> None:
    """Entry point for the ``thrindex`` and ``thx`` console scripts."""
    argv = sys.argv[1:]
    if not argv:
        _print_help()
        sys.exit(0)

    subcommand = argv[0]
    rest = argv[1:]

    if subcommand in ("-h", "--help"):
        _print_help()
    elif subcommand in ("run",):
        _cmd_run(rest)
    elif subcommand in ("doctor",):
        _cmd_doctor(rest)
    elif subcommand in ("targets",):
        _cmd_targets(rest)
    elif subcommand in ("--version", "-V"):
        print(_sdk_version())
    else:
        _die(
            f"Unknown subcommand: {subcommand!r}\n"
            "Run `thrindex --help` for usage."
        )


def _print_help() -> None:
    print(
        f"thrindex {_sdk_version()}\n"
        "\n"
        "Usage: thrindex <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  run <model.thx>            Run the behavioral simulator\n"
        "  run <model.thx> --input FILE   Run with spike input from a JSON file\n"
        "  doctor                     Diagnose the installation\n"
        "  targets                    List available simulation targets\n"
        "\n"
        "Run `thrindex <subcommand> --help` for subcommand options."
    )
