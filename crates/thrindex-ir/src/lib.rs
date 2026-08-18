//! Graph IR — the pre-lowering, continuous-parameter representation of SNN models.
//!
//! ## Design authority
//!
//! - **ADR-0008** (Graph IR time semantics: discrete timesteps with a declared `dt`):
//!   the [`GraphModel::dt_ms`] field is the canonical timestep; [`GraphLif`] holds
//!   continuous `tau_mem`/`tau_syn`, never the resolved `alpha`.
//! - **ADR-0009** (Synaptic delays: first-class, per-connection integer,
//!   capability-negotiated lowering): [`Delays`] is optional per connection-bearing
//!   layer; zero-delay models carry no delay bytes.
//!
//! ## Two-level parameter rule (ADR-0008, binding)
//!
//! The types in this crate enforce the rule **by construction**:
//! - [`GraphLif`] has `tau_mem` and `tau_syn` — **no `alpha` or `alpha_syn`**.
//!   Attempting to store a resolved `alpha` in the Graph IR is a **compile error**.
//! - [`GraphDense`] has optional [`Delays`] — **no `delay_steps`** resolved for any
//!   specific target `dt`. Resolved constants live only in the target-side artifact
//!   produced by `thrindex-compiler::lower`.
//!
//! ## Conv2d delay support: deferred (ADR-0009 v1 scope)
//!
//! [`GraphConv2d`] has **no** `delays` field. Conv2d-delay support is explicitly
//! deferred. Reopening trigger: a design partner ships a Conv2d-delay model where the
//! absence of this field is a proven constraint. Until then the field is absent and
//! unrepresentable, not merely unimplemented.

pub mod graph;

pub use graph::{Delays, GraphConv2d, GraphDense, GraphFlatten, GraphLayer, GraphLif, GraphModel, ResetMode};
