# Error Code Registry

Every user-facing error in thrindex carries a **stable `E####` code** (§30 / ENGINEERING.md).

Rules:
- Codes are **never reused** and **never renumbered**.
- Each error carries the four-part §30 body: What happened / Why / How to fix (cheapest first) / Docs link.
- All error messages are **snapshot-tested** — the string is a contract, not a style hope.

## M2 codes (E0001 – E0040)

| Code | Summary | Source |
|------|---------|--------|
| E0001 | `.thx` file not found | `thrindex-sim` |
| E0002 | Unsupported `.thx` format version | `thrindex-sim` |
| E0003 | Unknown layer type in model | `thrindex-sim` |
| E0004 | Layer dimension mismatch | `thrindex-sim` |
| E0005 | Invalid LIF parameters | `thrindex-sim` |
| E0006 | Base64 weight decode error | `thrindex-sim` |
| E0007 | Weight array length mismatch | `thrindex-sim` |
| E0008 | Artifact JSON parse error | `thrindex-sim` |
| E0009 | Artifact CRC32 integrity failure | `thrindex-sim` |
| E0010 | Input dimension mismatch | `thrindex-sim` |

## M3 codes (E0101 – E0109)

| Code | Summary | Source | Pass |
|------|---------|--------|------|
| E0101 | LIF `tau_mem` ≤ effective `dt` (authored or retimed) | `thrindex-compiler` | validate / lower |
| E0102 | Synaptic delay cannot be retimed — non-integer step ratio | `thrindex-compiler` | lower |
| E0103 | Synaptic delay exceeds target's declared maximum | `thrindex-compiler` | lower |
| E0104 | Target has no native delay support and emulation is disabled | `thrindex-compiler` | lower |
| E0105 | Model has no layers | `thrindex-compiler` | validate |
| E0106 | Consecutive layers have incompatible dimensions | `thrindex-compiler` | validate |
| E0107 | LIF `reset` string is not recognized (validated in validate pass, not serde) | `thrindex-compiler` | validate |
| E0108 | Graph IR JSON parse error | `thrindex-compiler` | capture |
| E0109 | Dense delay array length does not match `in_features × out_features` | `thrindex-compiler` | validate |

## Reserved ranges

| Range | Owner |
|-------|-------|
| E0001 – E0099 | M2 (sim + CLI) |
| E0100 – E0199 | M3 (compiler / Graph IR) |
| E0200 – E0299 | M4 (quantisation / fixed-point) |
| E0300 – E0399 | M5 (hardware targets) |

## Format

Every error message in production code must match:

```
E####: <one-line summary of what happened>
Why: <root cause>
Fix: <cheapest fix first>; <next option>
Docs: https://docs.thrindex.com/errors/E####
```

The Docs link may be a placeholder in early milestones — update when docs ship.
