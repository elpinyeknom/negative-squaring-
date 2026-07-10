# Negative Squaring: Pre-Tilting Weights to Preserve Reasoning in Quantized Models

**A preliminary toy-model study**
*Concept by elpinyeknom · Experiments run with Claude (Anthropic) · July 2026*

---

## Summary (the whole paper in one paragraph)

When AI models are compressed by rounding their internal numbers ("quantization"), simple chat survives but multi-step reasoning breaks, because tiny rounding errors snowball across a long chain of thought. This paper tests a new idea called **negative squaring**: before rounding, deliberately tilt the model's numbers *against* the direction the rounding damage will push its thinking — the way a golfer aims left because she knows the wind blows right. In a small simulated "toy brain," the method removed **77% of the reasoning error** caused by rounding and cut wrong final answers from **20 out of 50 down to 4–5 out of 50**, with zero increase in model size or running cost. The key discovery was that the tilt must be kept on a tight leash — never larger than half a rounding step — so that it only influences *which way borderline numbers round*, rather than distorting the numbers themselves.

---

## 1. The problem

Big AI models are like enormous recipe books where every ingredient is written with extreme precision. To fit one on a laptop, we round the numbers — the book shrinks dramatically, but every dish comes out slightly off.

For one-step tasks, "slightly off" doesn't matter. But reasoning is a 40-step wedding cake: each step's small error feeds the next step, and by the end the cake is leaning. This is why compressed models often still chat fluently yet stumble on math and logic.

Today's standard compression methods (the family that includes GPTQ and AWQ) fix errors **one layer at a time, looking one step ahead**. None of them ask: *what will this rounding decision do to the model's thinking 300 words from now?*

## 2. The idea: negative squaring

Instead of rounding and living with the damage, predict which way the rounding will push the model's long-run thinking, and lean the opposite way **before** rounding. The name reflects the move: take the coming error and apply its negative in advance, so the two cancel.

Golfer analogy: don't aim at the flag. Aim left, because you know the wind blows right. The "wrong" aim plus the wind lands the ball on the green.

## 3. How we tested it

Since we can't retrain a real 14-billion-number model on a laptop, we built a **toy brain**: 12 layers of connected numbers (about 49,000 weights) that "thinks" by passing a signal through itself **30 times in a row** — a stand-in for a 30-step chain of thought. We then:

1. Recorded the toy brain's exact thinking on practice questions (the answer key).
2. Rounded its weights to a coarse 4-bit grid (aggressive compression).
3. Searched for tiny pre-tilts to the original weights so that, *after* rounding, the compressed brain's full 30-step thinking matched the original as closely as possible.
4. Measured performance on **50 brand-new test questions** the tilt had never seen, including 20–50 yes/no decisions to check whether final answers flipped.

Three search strategies were compared: random trial-and-error, a "smart" search that traces blame backward through the whole thinking chain (gradient-based), and a combination of both.

## 4. Results

| Method | Reasoning error removed | Wrong final answers (of 50) |
|---|---|---|
| Naive rounding (today's baseline) | 0% | 20 |
| Random search, unleashed (v1) | ~18% | ~14-equivalent |
| Smart search, unleashed | *made things worse* | no better |
| Random search, leashed | 32% | 8 |
| **Smart search, leashed** | **77%** | **5** |
| Random + smart combo, leashed | 52% | **4** |

Three findings stand out:

**Finding 1 — it works.** The best versions removed most of the rounding damage to reasoning and cut wrong final answers by roughly 75%, at identical model size and identical running speed. The compressed brain isn't bigger or slower; it's just *aimed better*.

**Finding 2 — the golfer move emerged on its own.** In early runs, the pre-tilted brain was actually *worse* on its first thinking step, then pulled ahead by step 5 and halved the error by step 10. Nobody programmed that: the search independently discovered that sacrificing the opening move wins the long game — exactly the aim-left strategy the idea predicted.

**Finding 3 — the leash is everything.** Unconstrained, the smart search overshot and made reasoning worse. Constrained so no tilt may exceed **half a rounding step**, it became the best method. This reframes what negative squaring really is: not adjusting weight values, but **choosing which way each borderline weight rounds**, informed by the entire chain of thought rather than one step ahead. The tilt is a voting advisor for coin-flip rounding decisions.

## 4.5 The compression cliff: testing 4-bit, 3-bit, and 2-bit

Rerunning the entire experiment at three compression levels:

| Compression | Naive error | Pre-tilted error | Error removed | Wrong answers: naive → pre-tilted (of 50) |
|---|---|---|---|---|
| 4-bit | 0.0032 | 0.0007 | 77% | 20 → 5 |
| 3-bit | 0.0757 | 0.0034 | 95% | 33 → 15 |
| 2-bit | 0.1254 | 0.0399 | 68% | 20 → 19 (coin-flip territory) |

Three conclusions. First, pre-tilted 3-bit matches naive 4-bit on accuracy
and beats it on final decisions — the same reasoning quality in a model
roughly 25% smaller. Second, the method helps most exactly where compression
hurts most (95% removed at 3-bit). Third, it has a floor: at 2-bit, each
weight has only four possible values, and choosing the better rounding of a
ruined value cannot recover information that no longer exists. Negative
squaring moves the reasoning cliff from ~4-bit down to ~3-bit; it does not
abolish it.

## 5. Honest limitations

- **Scale.** The toy has 49,000 weights; real models have billions. The principle is demonstrated; the engineering is not.
- **Forgiving physics.** The toy brain naturally dampens its own errors over time. Real language models often *amplify* errors instead — which makes the problem more important there, and possibly the payoff larger, but this is untested.
- **Cost of the smart search.** Tracing blame backward through a full chain of thought is expensive at real scale. The leash helps enormously (it shrinks the search to near-tie rounding decisions only), but making this efficient for billion-weight models is the central unsolved challenge — and likely the reason no shipped method does this today.
- **Small calibration set.** Eight practice questions sufficed for the toy; real models would need calibration on genuine reasoning traces (math and logic chains), not generic text.

## 6. What it would take to scale

1. **Identify the borderline weights.** In any layer, only a small fraction of weights sit near a rounding boundary. Only those need the treatment — the search space collapses from billions to millions.
2. **Calibrate on reasoning, not chatter.** Use step-by-step math/logic transcripts as the practice questions.
3. **Cheap blame-tracing.** Approximate the backward trace with short chains first (5–10 steps), extending only where it pays off.
4. **Combine with existing tricks.** Negative squaring is compatible with residual corrections ("cubing"), outlier-friendly grids ("pi"), and mixed precision — they fix different failure modes and should stack.

## 7. Conclusion

Negative squaring — pre-tilting weights against their future rounding damage, constrained to half a rounding step — reduced reasoning errors by 77% and wrong answers by ~75% in a controlled toy model, at zero size or speed cost. The mechanism is best understood as **whole-chain-aware rounding**: making each borderline rounding decision with the entire chain of thought in view, where today's methods look only one step ahead. Scaling the blame-tracing search efficiently is the open problem standing between this principle and a shippable method for running capable reasoning models on consumer hardware like a 16GB MacBook Air.

---

*Experiment code: a 12-layer, 64-dimension recurrent toy network, 30-step trajectories, 4-bit per-row quantization, straight-through gradient search with tilt clipping at half the quantization step, evaluated on 50 held-out inputs. All results reproducible from the accompanying scripts.*
