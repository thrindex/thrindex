"""Build the frozen ratification fixture set for ADR-0010 Part II.

Produces ≥100 SHD test samples preprocessed **exactly** as train.py does,
stratified across all 20 classes, saved as versioned sparse-event JSON fixtures
with a manifest that records the fixture-set CRC32 for hash-pinned reproducibility.

Usage
-----
    uv run python scripts/freeze_shd_fixtures.py \\
        --data-dir /tmp/shd \\
        --out-dir  conformance/fixtures/shd_ratify_v1 \\
        --n-per-class 6

Output
------
    conformance/fixtures/shd_ratify_v1/
    ├── manifest.json           ← commit alongside fixtures
    └── frozen/
        ├── sample_000.json     ← sparse_events_v1 format
        ├── sample_001.json
        ...
        └── sample_NNN.json

Format (each sample_NNN.json)
------------------------------
    {
      "version": "sparse_events_v1",
      "label": 7,
      "T": 100,
      "N_in": 700,
      "events": [[0,42],[0,137],[1,200],...]
    }

Events are deduplicated (t, u) pairs after binning — identical to what
`torch.Tensor[i, t_idx, u_idx] = 1.0` produces when multiple spikes
land in the same (t, u) cell.

Preprocessing parameters (must match train.py exactly)
-------------------------------------------------------
    N_IN     = 700
    T        = 100
    max_time = 1.4  (seconds)
    dt       = max_time / T = 0.014 s per bin
    t_idx    = int(spike_time / dt), clipped to [0, T-1]
    u_idx    = int(unit_index),      clipped to [0, N_IN-1]

CRC32 fingerprint
-----------------
The manifest records the fixture-set fingerprint as computed by
`ratify_envelope`'s `fixture_fingerprint()` Rust function:
  1. For each file in index order: CRC32(file_bytes) as 8-char hex.
  2. Concatenate all 8-char strings.
  3. CRC32 of the concatenation → `crc32=XXXXXXXX (N files)`.

This fingerprint IS the link between any ratified threshold and the
exact frozen set that produced it (ADR-0010 Part II Step 1).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path


# ── Preprocessing constants (must be byte-identical to train.py) ──────────────

N_IN: int = 700
T: int = 100
MAX_TIME: float = 1.4   # seconds — SHD recording duration
DT: float = MAX_TIME / T  # 0.014 s per bin


# ── CRC32 helpers (must match Rust crc32fast::hash exactly) ──────────────────

def _crc32(data: bytes) -> int:
    """CRC32 of raw bytes, matching crc32fast::hash output (unsigned 32-bit)."""
    return zlib.crc32(data) & 0xFFFF_FFFF


def fixture_fingerprint(file_bytes_list: list[bytes]) -> str:
    """Compute the combined fixture-set CRC32 fingerprint.

    Algorithm matches ``ratify_envelope::fixture_fingerprint()`` exactly:
    1. CRC32 of each file's bytes → 8-char lowercase hex.
    2. Concatenate all hex strings.
    3. CRC32 of the concatenation → ``crc32=XXXXXXXX (N files)``.
    """
    combined = "".join(f"{_crc32(b):08x}" for b in file_bytes_list)
    set_crc = _crc32(combined.encode("utf-8"))
    return f"crc32={set_crc:08x} ({len(file_bytes_list)} files)"


# ── SHD loading ───────────────────────────────────────────────────────────────

def _bin_sample(spike_times: "np.ndarray", spike_units: "np.ndarray") -> list[list[int]]:
    """Convert raw SHD spike events into sorted deduplicated (t, u) pairs.

    Preprocessing is byte-identical to train.py::_load_shd():
        t_idx = int(spike_time / dt).clip(0, T-1)
        u_idx = int(unit_index).clip(0, N_IN-1)

    Multiple spikes landing in the same (t, u) bin produce a single event,
    because the downstream tensor uses ``spike_tensor[i, t_idx, u_idx] = 1.0``
    (a set, not an accumulate).
    """
    t_idx = (spike_times / DT).astype("int64").clip(0, T - 1)
    u_idx = spike_units.astype("int64").clip(0, N_IN - 1)
    # Deduplicate: encode each (t, u) pair as a single int, use set.
    encoded = set(int(t) * N_IN + int(u) for t, u in zip(t_idx, u_idx))
    events = sorted([int(e // N_IN), int(e % N_IN)] for e in encoded)
    return events


def load_shd_test(h5_path: Path) -> tuple[list, list[int]]:
    """Load raw SHD test-set spike events and labels from shd_test.h5.

    Returns:
        samples: list of (spike_times_array, spike_units_array) per sample.
        labels:  list of int class labels.
    """
    try:
        import h5py
    except ImportError:
        sys.exit("h5py required — run: uv add h5py")

    with h5py.File(h5_path, "r") as f:
        spike_times_all = f["spikes"]["times"][:]
        spike_units_all = f["spikes"]["units"][:]
        labels_raw = f["labels"][:]

    samples = list(zip(spike_times_all, spike_units_all))
    labels = [int(x) for x in labels_raw]
    return samples, labels


# ── Stratified selection ──────────────────────────────────────────────────────

def stratified_select(
    labels: list[int],
    n_per_class: int,
) -> list[int]:
    """Return indices of the first ``n_per_class`` samples from each class.

    Selection is by order of first appearance in the test set — deterministic
    and reproducible from the original h5 file (no shuffling).
    """
    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    n_classes = sorted(by_class.keys())
    missing = [c for c in n_classes if len(by_class[c]) < n_per_class]
    if missing:
        for c in missing:
            print(f"  WARNING: class {c} has only {len(by_class[c])} samples "
                  f"(requested {n_per_class}); taking all available.")

    selected: list[int] = []
    for cls in n_classes:
        selected.extend(by_class[cls][:n_per_class])

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen SHD ratification fixtures (ADR-0010 Part II Step 1)."
    )
    parser.add_argument(
        "--data-dir", default="/tmp/shd",
        help="Directory containing shd_test.h5 (default: /tmp/shd)."
    )
    parser.add_argument(
        "--out-dir", default="conformance/fixtures/shd_ratify_v1",
        help="Output directory for fixtures and manifest."
    )
    parser.add_argument(
        "--n-per-class", type=int, default=6,
        help="Samples per class (default: 6 → 120 total for 20 classes)."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    frozen_dir = out_dir / "frozen"
    h5_path = data_dir / "shd_test.h5"

    if not h5_path.exists():
        sys.exit(
            f"shd_test.h5 not found at {h5_path}.\n"
            "Run the SHD training script first:\n"
            "  uv run python templates/keyword-spotting/train.py "
            f"--data-dir {data_dir}"
        )

    frozen_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5_path} …")
    samples_raw, labels = load_shd_test(h5_path)
    n_total = len(labels)
    n_classes = len(set(labels))
    print(f"  {n_total} samples, {n_classes} classes")

    print(f"Stratified selection: {args.n_per_class} per class …")
    selected_indices = stratified_select(labels, args.n_per_class)
    print(f"  {len(selected_indices)} samples selected")

    # ── Write fixtures ─────────────────────────────────────────────────────────
    print(f"Writing fixtures to {frozen_dir} …")
    class_counts: dict[int, int] = collections.defaultdict(int)
    file_bytes_list: list[bytes] = []
    file_crcs: dict[str, str] = {}
    total_events = 0

    for out_idx, src_idx in enumerate(selected_indices):
        spike_times, spike_units = samples_raw[src_idx]
        label = labels[src_idx]
        events = _bin_sample(spike_times, spike_units)
        total_events += len(events)
        class_counts[label] += 1

        record = {
            "version": "sparse_events_v1",
            "label": label,
            "T": T,
            "N_in": N_IN,
            "events": events,
        }

        filename = f"sample_{out_idx:03d}.json"
        file_path = frozen_dir / filename

        # Write with compact separators (no spaces) to minimise file size.
        json_str = json.dumps(record, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")

        file_path.write_bytes(json_bytes)
        file_bytes_list.append(json_bytes)
        file_crcs[filename] = f"{_crc32(json_bytes):08x}"

        if (out_idx + 1) % 20 == 0:
            print(f"  wrote {out_idx + 1}/{len(selected_indices)}")

    print(f"  done. Total events: {total_events:,} "
          f"(avg {total_events/len(selected_indices):.0f} per sample)")

    # ── Compute fixture-set fingerprint (matches Rust fixture_fingerprint()) ──
    fingerprint = fixture_fingerprint(file_bytes_list)
    print(f"  Fixture-set fingerprint: {fingerprint}")

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest = {
        "version": "shd_ratify_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Frozen SHD test-set fixture set for ADR-0010 Part II ratification. "
            "Stratified across all 20 SHD classes. "
            "Do NOT modify these files — the fixture-set CRC32 is the link "
            "between any ratified threshold and the exact data that produced it."
        ),
        "preprocessing": {
            "source": "Spiking Heidelberg Digits (SHD) test split",
            "source_url": "https://zenkelab.org/resources/spiking-heidelberg-datasets/",
            "T": T,
            "N_in": N_IN,
            "max_time_s": MAX_TIME,
            "dt_s": DT,
            "binning": "t_idx = int(spike_time_s / dt_s), clipped to [0, T-1]; "
                       "u_idx = int(unit_index), clipped to [0, N_in-1]. "
                       "Identical to train.py::_load_shd().",
            "deduplication": "Multiple spikes in same (t, u) bin produce one event "
                             "(matches torch tensor assignment semantics).",
        },
        "selection": {
            "strategy": "stratified — first N per class by test-set index order",
            "n_per_class": args.n_per_class,
            "n_classes": n_classes,
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "source_indices": selected_indices,
        },
        "n_samples": len(selected_indices),
        "fixture_fingerprint": fingerprint,
        "file_crcs": file_crcs,
        "ratification_note": (
            "Use `cargo run -p conformance --bin ratify_envelope -- "
            f"--artifact templates/keyword-spotting/model.thx "
            f"--data-dir {out_dir}` to run the ratification measurement. "
            "Paste the full output into ADR-0010 Part II Amendment for founder approval."
        ),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"  Manifest written to {manifest_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f" Fixture set ready: {len(selected_indices)} samples, {n_classes} classes")
    print(f" Fingerprint: {fingerprint}")
    print()
    print(" Next step — run ratification measurement:")
    print()
    print("   cargo run -p conformance --bin ratify_envelope -- \\")
    print(f"     --artifact templates/keyword-spotting/model.thx \\")
    print(f"     --data-dir {out_dir}")
    print()
    print(" Commit both the fixtures and manifest:")
    print(f"   git add {out_dir}")
    print("   git commit -m 'feat(M4): freeze SHD ratification fixtures (ADR-0010 Part II Step 1)'")
    print("=" * 64)


if __name__ == "__main__":
    main()
