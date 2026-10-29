//! Two dense layers, no bias. The simulator float and the 4-bit grid.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use thrindex_compiler::compile;
use thrindex_sim::model::load_from_str;
use thrindex_sim::sim::{SimConfig, run};

/// `snn.Dense(..., bias=False)` emits `bias_b64: null`. Both layers use
/// integer weights whose maximum is 7, so the 4-bit scale is 1 and each
/// weight is its own code. The input keeps each hidden potential inside
/// 0..=15, which is the range a 4-bit activation can pass to the next layer.
#[test]
fn two_dense_layers_stay_on_the_four_bit_grid() {
    // Proven identity weights stay code 7. A 3 next to a max of 7 must stay
    // code 3. The old round(w / scale * 7) turned that 3 into 7.
    let identity = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let (ident_codes, ident_scale) = int4_codes(&identity);
    assert_eq!(ident_codes, vec![7, 0, 0, 0, 0, 7, 0, 0]);
    assert_eq!(ident_codes, collapsed_codes(&identity));
    assert!((ident_scale - (1.0 / 7.0)).abs() < 1e-12);

    let w1_f = [7.0f32, 0.0, 0.0, 0.0, 3.0, 1.0];
    let w2_f = [1.0f32, 4.0, 7.0, 0.0];
    let (c1, _) = int4_codes(&w1_f);
    assert_eq!(c1, vec![7, 0, 0, 0, 3, 1]);
    assert_ne!(c1, collapsed_codes(&w1_f));

    let w1 = f32_b64(&w1_f);
    let w2 = f32_b64(&w2_f);
    let ir = format!(
        r#"{{
          "dt_ms": 1.0,
          "layers": [
            {{"type":"dense","in_features":3,"out_features":2,"weights_b64":"{w1}","bias_b64":null,"delays":null}},
            {{"type":"dense","in_features":2,"out_features":2,"weights_b64":"{w2}","bias_b64":null,"delays":null}}
          ]
        }}"#
    );
    let report = compile(&ir, "sim").expect("two dense layers should compile");
    assert!(report.advisory.is_none());
    let artifact: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    assert_eq!(artifact["target"], "sim");
    assert_eq!(
        artifact["model"]["layers"][0]["bias_b64"],
        serde_json::Value::Null
    );
    assert_eq!(
        artifact["model"]["layers"][1]["bias_b64"],
        serde_json::Value::Null
    );
    assert_eq!(artifact["model"]["resource_report"]["input_shape"], 3);
    assert_eq!(artifact["model"]["resource_report"]["output_shape"], 2);
    assert_eq!(artifact["model"]["resource_report"]["lif_neuron_count"], 0);

    // Same decimals the device command will parse. 1/15 and 2/15 land on
    // input codes 1 and 2. Hidden potentials are 7 and 6, both under 15.
    let first = token(1.0 / 15.0);
    let second = token(2.0 / 15.0);
    let input = [parse_f32(&first), parse_f32(&second), parse_f32("0")];
    assert_eq!(input_code(input[0]), 1);
    assert_eq!(input_code(input[1]), 2);
    assert_eq!(input_code(input[2]), 0);

    let model = load_from_str(&report.thx_json).expect("sim loads the artifact");
    let output = run(&model, &[vec![input.to_vec()]], &SimConfig { threads: 1 })
        .expect("sim runs one frame");
    let sim = &output.spikes[0][0];
    let grid = four_bit_dense(&input, &w1_f, 3, &w2_f, 2);

    assert_eq!(sim.len(), 2);
    assert_eq!(grid.len(), 2);
    for (got, expected) in sim.iter().zip(grid.iter()) {
        let ulps = (got.to_bits() as i32).abs_diff(expected.to_bits() as i32);
        assert!(
            ulps <= 1,
            "simulator {got} and 4-bit grid {expected} differ by {ulps} ulps"
        );
    }

    eprintln!("INPUT {first} {second} 0");
    eprintln!("SIM {} {}", sim[0], sim[1]);
    eprintln!("GRID {} {}", grid[0], grid[1]);

    if let Ok(path) = std::env::var("WRITE_TWO_DENSE_THX") {
        std::fs::write(&path, &report.thx_json).expect("write two-dense artifact");
    }
}

fn f32_b64(values: &[f32]) -> String {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    B64.encode(raw)
}

fn token(value: f32) -> String {
    let text = format!("{value}");
    assert_eq!(
        parse_f32(&text),
        value,
        "stdin token {text} must parse back to the same f32"
    );
    text
}

fn parse_f32(text: &str) -> f32 {
    text.parse().expect("f32")
}

fn input_code(value: f32) -> i32 {
    let scaled = f64::from(value) * 15.0;
    let code = scaled.round();
    assert!((0.0..=15.0).contains(&code));
    assert_eq!(code as i32, (value * 15.0).round() as i32);
    code as i32
}

/// Integer path for a linear last layer on the 4-bit grid.
///
/// Each layer's scale is `max(|W|) / 7`. Hidden potentials must already lie
/// in `0..=15`; a 4-bit activation clips anything outside that, and this
/// input is chosen so the clip does not move the value. The returned floats
/// are `potential * scale_1 * scale_2 / 15`, packed the way the program stores
/// the output scale (f32).
fn four_bit_dense(input: &[f32], w1: &[f32], in1: usize, w2: &[f32], in2: usize) -> Vec<f32> {
    let (c1, s1) = int4_codes(w1);
    let (c2, s2) = int4_codes(w2);
    let x: Vec<i32> = input.iter().copied().map(input_code).collect();
    let hidden = matmul_i32(&x, &c1, in1);
    for potential in &hidden {
        assert!(
            (0..=15).contains(potential),
            "hidden potential {potential} would be clipped by a 4-bit activation"
        );
    }
    let potential = matmul_i32(&hidden, &c2, in2);
    let scale = ((s1 * s2) / 15.0) as f32;
    potential.iter().map(|p| (*p as f32) * scale).collect()
}

fn int4_codes(weights: &[f32]) -> (Vec<i32>, f64) {
    let max_abs = weights
        .iter()
        .fold(0.0f64, |m, w| m.max(f64::from(*w).abs()));
    assert!(max_abs > 0.0);
    let scale = max_abs / 7.0;
    let codes = weights
        .iter()
        .map(|w| {
            let scaled = f64::from(*w) / scale;
            let rounded = scaled.round();
            assert!(
                (scaled - rounded).abs() < 1e-6,
                "weight {w} is not an exact 4-bit code"
            );
            let code = rounded as i32;
            assert!((-7..=7).contains(&code));
            code
        })
        .collect();
    (codes, scale)
}

/// `round(w / scale * 7)` then clip. Scale is already `max/7`, so this
/// saturates every non-zero weight. Kept here so the two-layer weights fail
/// it and the identity weights still pass it.
fn collapsed_codes(weights: &[f32]) -> Vec<i32> {
    let max_abs = weights
        .iter()
        .fold(0.0f64, |m, w| m.max(f64::from(*w).abs()));
    let scale = max_abs / 7.0;
    weights
        .iter()
        .map(|w| (f64::from(*w) / scale * 7.0).round().clamp(-7.0, 7.0) as i32)
        .collect()
}

fn matmul_i32(input: &[i32], codes: &[i32], width: usize) -> Vec<i32> {
    assert_eq!(codes.len() % width, 0);
    assert_eq!(input.len(), width);
    codes
        .chunks(width)
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}
