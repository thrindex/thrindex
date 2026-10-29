//! Resource summary — derived from model layer shapes.
//!
//! Older `.thx` files omit `resource_report`. The summary is then derived
//! from the layers. A file that includes `resource_report` must match this
//! derivation; see [`check_report`].

use crate::error::ArtifactError;

/// Geometry of a model, either derived from its layers or declared by the compiler.
///
/// Field order is the on-disk order inside `model.resource_report`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSummary {
    /// Total number of layers.
    pub layer_count: usize,
    /// Sum of LIF output neuron counts, derived from preceding Dense/Conv2d
    /// layer `out_features` / `out_channels`.
    pub lif_neuron_count: usize,
    /// Total number of trainable weights (Dense + Conv2d weight tensors),
    /// not including biases.
    pub total_weight_count: usize,
    /// Whether any Dense layer carries axonal delays (ADR-0009).
    pub has_delays: bool,
    /// Ordered list of layer type strings, e.g. `["dense", "lif", "dense", "lif"]`.
    pub layer_types: Vec<String>,
    /// Input dimension derived from the first Dense/Conv2d layer's input size.
    /// `None` if the first layer is a LIF (malformed model).
    pub input_shape: Option<usize>,
    /// Output dimension derived from the last Dense/Conv2d layer's output size.
    /// `None` if the last layer is a LIF.
    pub output_shape: Option<usize>,
}

/// Derive a [`ResourceSummary`] from the ordered slice of raw layer values.
///
/// This is the single geometry walk. The compiler stores the result, and
/// parse checks a stored report against a fresh call.
pub fn summarize_layers(layers: &[serde_json::Value]) -> ResourceSummary {
    derive(layers)
}

/// Reject a stored report that disagrees with the layers.
pub(crate) fn check_report(
    declared: &ResourceSummary,
    layers: &[serde_json::Value],
) -> Result<(), ArtifactError> {
    let derived = derive(layers);
    if declared == &derived {
        return Ok(());
    }
    Err(ArtifactError::InvalidContent {
        detail: format!(
            "resource_report does not match the layers ({})",
            report_diff(declared, &derived)
        ),
    })
}

fn report_diff(declared: &ResourceSummary, derived: &ResourceSummary) -> String {
    let mut parts = Vec::new();
    push_diff(
        &mut parts,
        "input_shape",
        &declared.input_shape,
        &derived.input_shape,
    );
    push_diff(
        &mut parts,
        "output_shape",
        &declared.output_shape,
        &derived.output_shape,
    );
    push_diff(
        &mut parts,
        "layer_count",
        &declared.layer_count,
        &derived.layer_count,
    );
    push_diff(
        &mut parts,
        "lif_neuron_count",
        &declared.lif_neuron_count,
        &derived.lif_neuron_count,
    );
    push_diff(
        &mut parts,
        "total_weight_count",
        &declared.total_weight_count,
        &derived.total_weight_count,
    );
    push_diff(
        &mut parts,
        "has_delays",
        &declared.has_delays,
        &derived.has_delays,
    );
    push_diff(
        &mut parts,
        "layer_types",
        &declared.layer_types,
        &derived.layer_types,
    );
    if parts.is_empty() {
        "declared report differs".to_string()
    } else {
        parts.join("; ")
    }
}

fn push_diff<T: std::fmt::Debug + PartialEq>(
    parts: &mut Vec<String>,
    name: &str,
    declared: &T,
    derived: &T,
) {
    if declared != derived {
        parts.push(format!(
            "{name} is {declared:?} but the layers describe {derived:?}"
        ));
    }
}

/// Derive a [`ResourceSummary`] from the ordered slice of raw layer `Value`s.
///
/// Unknown layer types are counted but do not contribute to weight / neuron
/// totals; `layer_types` will contain their `"type"` string (or `"unknown"`).
fn derive(layers: &[serde_json::Value]) -> ResourceSummary {
    let mut lif_neuron_count = 0usize;
    let mut total_weight_count = 0usize;
    let mut has_delays = false;
    let mut layer_types = Vec::with_capacity(layers.len());
    let mut last_output: usize = 0;

    for v in layers {
        let layer_type = v["type"].as_str().unwrap_or("unknown");
        layer_types.push(layer_type.to_string());

        match layer_type {
            "dense" => {
                let in_f = v["in_features"].as_u64().unwrap_or(0) as usize;
                let out_f = v["out_features"].as_u64().unwrap_or(0) as usize;
                total_weight_count += in_f * out_f;
                if v["delays_b64"].is_string() {
                    has_delays = true;
                }
                last_output = out_f;
            }
            "lif" => {
                lif_neuron_count += last_output;
            }
            "conv2d" => {
                let in_ch = v["in_channels"].as_u64().unwrap_or(0) as usize;
                let out_ch = v["out_channels"].as_u64().unwrap_or(0) as usize;
                let kh = v["kernel_h"].as_u64().unwrap_or(0) as usize;
                let kw = v["kernel_w"].as_u64().unwrap_or(0) as usize;
                total_weight_count += in_ch * out_ch * kh * kw;
                last_output = out_ch;
            }
            _ => {}
        }
    }

    let input_shape = layers.first().and_then(|v| match v["type"].as_str() {
        Some("dense") => v["in_features"].as_u64().map(|n| n as usize),
        Some("conv2d") => v["in_channels"].as_u64().map(|n| n as usize),
        _ => None,
    });

    let output_shape = layers.iter().rev().find_map(|v| match v["type"].as_str() {
        Some("dense") => v["out_features"].as_u64().map(|n| n as usize),
        Some("conv2d") => v["out_channels"].as_u64().map(|n| n as usize),
        _ => None,
    });

    ResourceSummary {
        layer_count: layers.len(),
        lif_neuron_count,
        total_weight_count,
        has_delays,
        layer_types,
        input_shape,
        output_shape,
    }
}
