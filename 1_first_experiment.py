import numpy as np

rng = np.random.default_rng(42)

# ---- Toy "brain": a chain of 12 layers, each mixing a 64-number state ----
DEPTH = 12
DIM = 64
STEPS = 30          # how many "thinking steps" (like tokens in a chain of thought)
BITS = 4            # 4-bit rounding, like aggressive quantization

layers = [rng.normal(0, 1/np.sqrt(DIM), (DIM, DIM)) for _ in range(DEPTH)]

def act(x):
    return np.tanh(x)  # gentle squashing, like a real network

def forward(ws, x):
    """One 'thinking step': pass state through all layers."""
    for W in ws:
        x = act(W @ x)
    return x

def reason(ws, x0, steps=STEPS):
    """A chain of thought: feed the output back in, over and over."""
    x = x0.copy()
    traj = []
    for _ in range(steps):
        x = forward(ws, x)
        traj.append(x.copy())
    return np.array(traj)

def quantize(W, bits=BITS):
    """Round each weight to a coarse grid (per-row scaling, like real methods)."""
    scale = np.abs(W).max(axis=1, keepdims=True) / (2**(bits-1) - 1)
    return np.round(W / scale) * scale

# ---- Baseline: quantize naively ----
q_layers = [quantize(W) for W in layers]

# ---- "Negative square": pre-tilt weights so that AFTER rounding, the
# ---- full chain-of-thought trajectory matches the original as closely
# ---- as possible. We nudge weights in the opposite direction of the
# ---- error the rounding will cause downstream, then re-round.
def trajectory_error(ws, ref_traj, x0s):
    err = 0.0
    for x0, ref in zip(x0s, ref_traj):
        traj = reason(ws, x0)
        err += np.mean((traj - ref)**2)
    return err / len(x0s)

# Calibration inputs ("practice questions")
N_CAL = 8
x0s = [rng.normal(0, 1, DIM) for _ in range(N_CAL)]
ref_trajs = [reason(layers, x0) for x0 in x0s]

def eval_ws(ws):
    return trajectory_error(ws, ref_trajs, x0s)

base_err = eval_ws(q_layers)

# Simple coordinate-free optimizer: random pre-tilt proposals, keep improvements.
# The tilt is applied to the FULL-precision weights, THEN we round.
# This is the essence of "aim left because the wind blows right."
tilts = [np.zeros_like(W) for W in layers]
best_err = base_err
sigma = 0.02
improved = 0
for it in range(400):
    li = rng.integers(0, DEPTH)
    proposal = tilts[li] + rng.normal(0, sigma, layers[li].shape)
    trial = [quantize(layers[i] + (proposal if i == li else tilts[i])) for i in range(DEPTH)]
    e = eval_ws(trial)
    if e < best_err:
        best_err = e
        tilts[li] = proposal
        improved += 1

tilted_layers = [quantize(layers[i] + tilts[i]) for i in range(DEPTH)]

# ---- Held-out test: fresh questions the tilt never saw ----
test_x0s = [rng.normal(0, 1, DIM) for _ in range(20)]
def test_err(ws):
    e = 0.0
    for x0 in test_x0s:
        ref = reason(layers, x0)
        traj = reason(ws, x0)
        e += np.mean((traj - ref)**2)
    return e / len(test_x0s)

def drift_curve(ws):
    """Average error at each thinking step (does error compound?)"""
    curves = []
    for x0 in test_x0s:
        ref = reason(layers, x0)
        traj = reason(ws, x0)
        curves.append(np.mean((traj - ref)**2, axis=1))
    return np.mean(curves, axis=0)

naive_test = test_err(q_layers)
tilt_test = test_err(tilted_layers)

naive_curve = drift_curve(q_layers)
tilt_curve = drift_curve(tilted_layers)

# ---- Also test a "final answer" flip rate: does the sign of a readout
# ---- (like a yes/no decision) flip vs the original model?
readout = rng.normal(0, 1, DIM)
def decision_flips(ws):
    flips = 0
    for x0 in test_x0s:
        ref = reason(layers, x0)[-1] @ readout
        got = reason(ws, x0)[-1] @ readout
        flips += (np.sign(ref) != np.sign(got))
    return flips

print(f"Calibration error  naive: {base_err:.5f}   pre-tilted: {best_err:.5f}   ({improved} improvements accepted)")
print(f"Held-out test error naive: {naive_test:.5f}   pre-tilted: {tilt_test:.5f}   reduction: {100*(1-tilt_test/naive_test):.1f}%")
print(f"Decision flips (out of 20): naive {decision_flips(q_layers)}  pre-tilted {decision_flips(tilted_layers)}")
print()
print("Error growth over thinking steps (naive vs tilted):")
for s in [0, 4, 9, 19, 29]:
    print(f"  step {s+1:2d}:  naive {naive_curve[s]:.5f}   tilted {tilt_curve[s]:.5f}")
