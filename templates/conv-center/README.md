# Template: 5×5 center-tap convolution

One `conv2d` layer. No LIF. No bias. Not a trained product model.

The kernel is 3×3 with a 1 at the center and zeros elsewhere. Padding is 1,
so the map stays 5×5. `input_height` and `input_width` are in the file.
A `0.2` at the center pixel (index 12 in row-major order) comes back from
the simulator as `0.2`. The other 24 values come back as `0`.
