# Template: two dense layers

Two `dense` layers. No LIF. No bias. Not a trained product model.

The first layer is 3→2:

```
7  0  0
0  3  1
```

The second layer is 2→2:

```
1  4
7  0
```

Each layer's largest weight is 7, so the 4-bit scale is 1 and each weight
is its own code. The input `0.06666667 0.13333334 0` is codes 1, 2, and 0.
The hidden potentials are 7 and 6, which sit inside 0..=15. The simulator
returns:

```
2.0666668
3.266667
```
