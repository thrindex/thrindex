//! Spatial frames for `Conv2d` and `Flatten`.
//!
//! A frame is either a flat feature vector or a channel-height-width image with
//! no batch axis. Flatten interprets that image as a batch of one, so
//! `start_dim` / `end_dim` mean the same thing as `torch.nn.Flatten` on
//! `[1, C, H, W]`.

use crate::error::SimError;
use crate::model::{Conv2dLayer, FlattenLayer};

/// One layer's activation inside a single sample.
#[derive(Debug, Clone)]
pub(crate) struct Activation {
    pub values: Vec<f32>,
    /// `None` is a flat vector. `Some` is `(channels, height, width)`.
    pub spatial: Option<(usize, usize, usize)>,
}

impl Activation {
    pub(crate) fn flat(values: Vec<f32>) -> Self {
        Self {
            values,
            spatial: None,
        }
    }

    /// The flat feature vector a `Dense` or the final raster expects.
    pub(crate) fn as_features(&self, layer_idx: usize) -> Result<&[f32], SimError> {
        if self.spatial.is_some() {
            return Err(SimError::SpatialShape {
                layer_idx,
                message: "Dense received a spatial frame".into(),
                reason: "a fully connected layer reads a vector, not an image".into(),
                fix: "place snn.Flatten() between the convolution and the dense layer".into(),
            });
        }
        Ok(&self.values)
    }
}

/// Cross-correlation matching `torch.nn.functional.conv2d`.
///
/// Returns the output frame and the number of in-bounds spike multiplies.
pub(crate) fn conv2d(
    input: &Activation,
    layer: &Conv2dLayer,
    layer_idx: usize,
) -> Result<(Activation, u64), SimError> {
    let (channels, height, width) = spatial_of(input, layer, layer_idx)?;
    if channels != layer.in_channels {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: format!(
                "Conv2d expected {} input channels and received {channels}",
                layer.in_channels
            ),
            reason: "the frame reaching this layer has a different channel count".into(),
            fix: "recompile the stack so each convolution's in_channels matches the previous layer"
                .into(),
        });
    }
    let out_h = out_size(
        height,
        layer.padding[0],
        layer.dilation[0],
        layer.kernel_h,
        layer.stride[0],
        layer_idx,
        "height",
    )?;
    let out_w = out_size(
        width,
        layer.padding[1],
        layer.dilation[1],
        layer.kernel_w,
        layer.stride[1],
        layer_idx,
        "width",
    )?;
    let in_per_group = layer.in_channels / layer.groups;
    let out_per_group = layer.out_channels / layer.groups;
    let mut output = vec![0.0_f32; layer.out_channels * out_h * out_w];
    if let Some(bias) = &layer.bias {
        for oc in 0..layer.out_channels {
            let fill = bias[oc];
            for yx in 0..(out_h * out_w) {
                output[oc * out_h * out_w + yx] = fill;
            }
        }
    }
    let mut ops = 0_u64;
    for oc in 0..layer.out_channels {
        let group = oc / out_per_group;
        let channel_base = group * in_per_group;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = output[oc * out_h * out_w + oy * out_w + ox];
                for ic in 0..in_per_group {
                    for ky in 0..layer.kernel_h {
                        for kx in 0..layer.kernel_w {
                            let iy = oy * layer.stride[0] + ky * layer.dilation[0];
                            let ix = ox * layer.stride[1] + kx * layer.dilation[1];
                            let y_in = iy.checked_sub(layer.padding[0]);
                            let x_in = ix.checked_sub(layer.padding[1]);
                            let (Some(y_in), Some(x_in)) = (y_in, x_in) else {
                                continue;
                            };
                            if y_in >= height || x_in >= width {
                                continue;
                            }
                            let in_c = channel_base + ic;
                            let sample = input.values[(in_c * height + y_in) * width + x_in];
                            let weight =
                                layer.weights[(((oc * in_per_group + ic) * layer.kernel_h + ky)
                                    * layer.kernel_w)
                                    + kx];
                            acc += weight * sample;
                            if sample > 0.5 {
                                ops += 1;
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + oy * out_w + ox] = acc;
            }
        }
    }
    Ok((
        Activation {
            values: output,
            spatial: Some((layer.out_channels, out_h, out_w)),
        },
        ops,
    ))
}

/// `torch.nn.Flatten` on a leading batch of one.
pub(crate) fn flatten(
    input: &Activation,
    layer: &FlattenLayer,
    layer_idx: usize,
) -> Result<Activation, SimError> {
    let mut dims = match input.spatial {
        Some((c, h, w)) => vec![1, c, h, w],
        None => vec![1, input.values.len()],
    };
    let rank = i32::try_from(dims.len()).unwrap_or(i32::MAX);
    let start = normalize_dim(layer.start_dim, rank, layer_idx)?;
    let end = normalize_dim(layer.end_dim, rank, layer_idx)?;
    if start > end {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: format!("flatten start_dim={start} is after end_dim={end}"),
            reason: "torch.nn.Flatten requires the start dimension to precede the end".into(),
            fix: "set start_dim and end_dim the way nn.Flatten expects them".into(),
        });
    }
    let product = dims[start..=end]
        .iter()
        .try_fold(1_usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| SimError::SpatialShape {
            layer_idx,
            message: "flattened dimension overflowed".into(),
            reason: "the product of the flattened axes does not fit in a pointer-sized integer"
                .into(),
            fix: "reduce the spatial size of the frame".into(),
        })?;
    let mut next = Vec::new();
    next.extend_from_slice(&dims[..start]);
    next.push(product);
    next.extend_from_slice(&dims[end + 1..]);
    dims = next;
    if dims.len() == 2 && dims[0] == 1 {
        return Ok(Activation::flat(input.values.clone()));
    }
    if dims.len() == 4 && dims[0] == 1 {
        return Ok(Activation {
            values: input.values.clone(),
            spatial: Some((dims[1], dims[2], dims[3])),
        });
    }
    Err(SimError::SpatialShape {
        layer_idx,
        message: format!("flatten produced rank {}", dims.len()),
        reason: "the simulator runs a flat vector or a single image, not an arbitrary reshape"
            .into(),
        fix: "flatten from the first non-batch dimension through the end (the nn.Flatten default)"
            .into(),
    })
}

fn spatial_of(
    input: &Activation,
    layer: &Conv2dLayer,
    layer_idx: usize,
) -> Result<(usize, usize, usize), SimError> {
    if let Some(shape) = input.spatial {
        return Ok(shape);
    }
    let (Some(height), Some(width)) = (layer.input_height, layer.input_width) else {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: "Conv2d has no image size".into(),
            reason: "the first convolution needs the height and width of the frame it reads".into(),
            fix: "construct snn.Conv2d(..., input_height=H, input_width=W) and recompile".into(),
        });
    };
    let expected = layer.in_channels * height * width;
    if input.values.len() != expected {
        return Err(SimError::InputDimensionMismatch {
            expected,
            got: input.values.len(),
        });
    }
    Ok((layer.in_channels, height, width))
}

fn out_size(
    input: usize,
    pad: usize,
    dilation: usize,
    kernel: usize,
    stride: usize,
    layer_idx: usize,
    axis: &str,
) -> Result<usize, SimError> {
    if stride == 0 || dilation == 0 || kernel == 0 {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: format!(
                "Conv2d {axis} has stride={stride}, dilation={dilation}, kernel={kernel}"
            ),
            reason: "those convolution parameters must be at least 1".into(),
            fix: "recompile from a snn.Conv2d with positive kernel, stride, and dilation".into(),
        });
    }
    let input = i64::try_from(input).unwrap_or(i64::MAX);
    let pad = i64::try_from(pad).unwrap_or(i64::MAX);
    let span = i64::try_from(dilation.saturating_mul(kernel - 1)).unwrap_or(i64::MAX);
    let numer = input + 2 * pad - span - 1;
    if numer < 0 {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: format!("Conv2d {axis} output size is not positive"),
            reason: "the kernel does not fit in the frame with this padding, stride, and dilation"
                .into(),
            fix: "reduce the kernel or increase padding, matching a torch.nn.Conv2d that runs"
                .into(),
        });
    }
    let numer = usize::try_from(numer).unwrap_or(usize::MAX);
    Ok(numer / stride + 1)
}

fn normalize_dim(dim: i32, rank: i32, layer_idx: usize) -> Result<usize, SimError> {
    let shifted = if dim < 0 { dim + rank } else { dim };
    if shifted < 0 || shifted >= rank {
        return Err(SimError::SpatialShape {
            layer_idx,
            message: format!("flatten dimension {dim} is outside rank {rank}"),
            reason: "start_dim and end_dim are indexed the way torch.nn.Flatten indexes them"
                .into(),
            fix: "use the default Flatten(), which flattens every dimension after the batch".into(),
        });
    }
    Ok(usize::try_from(shifted).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Conv2dLayer;

    #[test]
    fn one_by_one_conv_scales_each_pixel() {
        let layer = Conv2dLayer {
            in_channels: 1,
            out_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride: [1, 1],
            padding: [0, 0],
            groups: 1,
            dilation: [1, 1],
            input_height: Some(2),
            input_width: Some(2),
            weights: vec![2.0],
            bias: None,
        };
        let input = Activation::flat(vec![1.0, 0.0, 0.5, 4.0]);
        let (out, ops) = conv2d(&input, &layer, 0).unwrap();
        assert_eq!(out.values, vec![2.0, 0.0, 1.0, 8.0]);
        assert_eq!(ops, 2);
        assert_eq!(out.spatial, Some((1, 2, 2)));
    }
}
