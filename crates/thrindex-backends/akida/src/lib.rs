//! BrainChip AKD1500 backend for THRINDEX.
//!
//! Implements [`thrindex_backend_api::Backend`] for the AKD1500 neuromorphic
//! co-processor (Akida 1.0). See RFC-004 / ADR-0011 for the full design rationale.
//!
//! ## Two-tier architecture (ADR-0011 Option C)
//!
//! **Tier 1** — Python compiler (`python/akida_compile.py`), runs on developer
//! machine with MetaTF installed:  `.thx` → `.fbz`
//!
//! **Tier 2** — This crate + `akida-runtime` binary (item 9), runs on the
//! Raspberry Pi beside the AKD1500:  `.fbz` → inference output
//!
//! ## What this backend accepts
//!
//! Only `.thx` artifacts compiled with `target: "akida-akd1500"` containing
//! **only** `Dense` or `Conv2d` layers, **no** `Lif` layers, and **no**
//! synaptic delays. See error codes E0401–E0407 in [`error`].
//!
//! ## Feature flags
//!
//! | Flag       | Effect                                                              |
//! |------------|---------------------------------------------------------------------|
//! | (default)  | Validation + capability only; `run_batch` returns E0204 stub       |
//! | `hardware` | Compiles Engine Library FFI + `akida-runtime` binary; `run_batch`  |
//! |            | spawns the binary via subprocess protocol                           |
//!
//! ## Error codes
//!
//! | Error | Trigger                                                    |
//! |-------|------------------------------------------------------------|
//! | E0401 | `Lif` layer in artifact                                    |
//! | E0402 | Synaptic delay in artifact                                 |
//! | E0403 | NaN / Inf weight in artifact                               |
//! | E0404 | Wrong target in artifact                                   |
//! | E0407 | `T > 1` timesteps in `run_batch` input                     |

pub mod backend;
pub mod capability;
pub mod error;
pub mod validate;

/// Engine Library FFI bridge (compiled only when `hardware` feature is enabled
/// AND `THRINDEX_AKIDA_ENGINE_PATH` was set at build time).
#[cfg(all(feature = "hardware", akida_engine_available))]
pub mod ffi;

pub use backend::AkidaBackend;
pub use capability::{AKD1500_TARGET_NAME, AKD1500_WEIGHT_MAX, akd1500_capability};
pub use error::AkidaError;
pub use validate::validate_artifact;
