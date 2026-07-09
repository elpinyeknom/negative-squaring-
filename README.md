# Negative Squaring — Toy Experiments

Code accompanying the writeup "Negative Squaring: Pre-Tilting Weights to
Preserve Reasoning in Quantized Models" (July 2026). See
`negative-squaring-paper.md` in this repository for the full plain-language paper.

## The idea in one line

Before quantizing a model, tilt each weight *against* the error the rounding
will cause across the model's whole multi-step reasoning trajectory — clipped
to half a quantization step, so the tilt only decides which way borderline
weights round.

## What's here

| File | What it does | Key result |
|---|---|---|
| `1_first_experiment.py` | Random-search pre-tilt vs naive 4-bit quantization on a 12-layer, 30-step recurrent toy network | ~18% trajectory error removed; decision flips 14/20 → 8/20 |
| `2_gradient_attempt.py` | Straight-through gradient search, unconstrained | Backfires — test error gets worse (documented negative result) |
| `3_final_with_clipping.py` | Gradient + random + combo searches, with tilts clipped to half a quantization step | 77% error removed; decision flips 20/50 → 4-5/50 |

## Run it

Requires only Python 3 and numpy:

```
pip install numpy
python 3_final_with_clipping.py
```

Each script is self-contained, seeded, and reproduces the numbers in the
writeup. Runtime is seconds to a few minutes on any laptop.

## Honest limitations

- Toy scale: ~49k weights, tanh recurrence, not a transformer.
- The toy's dynamics dampen errors; real LLMs often amplify them. Untested there.
- Full-trajectory backprop is expensive at real scale; the clipping constraint
  shrinks the search space (only near-boundary weights matter) but efficient
  scaling is unsolved.

## Open invitation

If you have compute and want to try trajectory-aware rounding on a real
sub-1B model, or you know prior literature that already does this
(AdaRound optimizes rounding decisions per-layer; we're looking for
*whole-trajectory* versions), please reach out in the thread or open an
issue here.
