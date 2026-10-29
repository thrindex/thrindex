# Template: dense feedforward (no LIF)

A `.thx` file with a single `Dense(4→2)` layer and no LIF. Not a trained
product model. Weights are a 4→2 slice of identity so a synthetic all-ones
input produces a non-zero activation.

Regenerate:

```sh
WRITE_DENSE_FEEDFORWARD_THX=templates/dense-feedforward/model.thx \
  cargo test -p thrindex-compiler compile_dense_only_succeeds
```
