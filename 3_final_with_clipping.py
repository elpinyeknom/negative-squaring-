import numpy as np

rng = np.random.default_rng(42)
DEPTH, DIM, STEPS, BITS = 12, 64, 30, 4
layers = [rng.normal(0, 1/np.sqrt(DIM), (DIM, DIM)) for _ in range(DEPTH)]

def scales(W, bits=BITS):
    return np.abs(W).max(axis=1, keepdims=True) / (2**(bits-1) - 1)

def quantize(W, bits=BITS):
    s = scales(W, bits)
    return np.round(W / s) * s

def reason(ws, x0, steps=STEPS):
    x = x0.copy(); traj = []
    for _ in range(steps):
        for W in ws:
            x = np.tanh(W @ x)
        traj.append(x.copy())
    return np.array(traj)

N_CAL = 8
x0s = [rng.normal(0, 1, DIM) for _ in range(N_CAL)]
refs = [reason(layers, x0) for x0 in x0s]
test_x0s = [rng.normal(0, 1, DIM) for _ in range(50)]  # bigger test set, less noise
test_refs = [reason(layers, x0) for x0 in test_x0s]
readout = rng.normal(0, 1, DIM)

def eval_test(ws):
    return np.mean([np.mean((reason(ws, x0) - r)**2) for x0, r in zip(test_x0s, test_refs)])

def flips(ws):
    return sum(np.sign(r[-1] @ readout) != np.sign(reason(ws, x0)[-1] @ readout)
               for x0, r in zip(test_x0s, test_refs))

def calib_err(ws):
    return np.mean([np.mean((reason(ws, x0) - r)**2) for x0, r in zip(x0s, refs)])

q_naive = [quantize(W) for W in layers]
print(f"Naive 4-bit: test {eval_test(q_naive):.5f} | flips {flips(q_naive)}/50")

# The key constraint: |tilt| <= half a rounding step. That's exactly enough
# to flip borderline rounding decisions, never enough to distort a weight.
half_step = [scales(W).repeat(DIM, axis=1) * 0.5 for W in layers]

def forward_record(ws, x0):
    x = x0.copy(); xs_in = []; zs = []; traj = []
    for t in range(STEPS):
        for l in range(DEPTH):
            xs_in.append(x); z = ws[l] @ x; zs.append(z); x = np.tanh(z)
        traj.append(x.copy())
    return np.array(traj), xs_in, zs

def gradients(ws, x0, ref):
    traj, xs_in, zs = forward_record(ws, x0)
    grads = [np.zeros_like(W) for W in ws]
    dx = np.zeros(DIM); idx = STEPS * DEPTH - 1
    for t in reversed(range(STEPS)):
        dx = dx + (2.0 / (DIM * STEPS)) * (traj[t] - ref[t])
        for l in reversed(range(DEPTH)):
            dz = dx * (1 - np.tanh(zs[idx])**2)
            grads[l] += np.outer(dz, xs_in[idx])
            dx = ws[l].T @ dz
            idx -= 1
    return grads

tilts = [np.zeros_like(W) for W in layers]
m = [np.zeros_like(W) for W in layers]; v = [np.zeros_like(W) for W in layers]
lr, b1, b2, eps = 2e-4, 0.9, 0.999, 1e-8
best = (calib_err(q_naive), [W.copy() for W in q_naive])
t_step = 0
for epoch in range(150):
    ws_eff = [quantize(layers[i] + tilts[i]) for i in range(DEPTH)]
    ce = calib_err(ws_eff)
    if ce < best[0]:
        best = (ce, [W.copy() for W in ws_eff])
    g_acc = [np.zeros_like(W) for W in layers]
    for x0, ref in zip(x0s, refs):
        gs = gradients(ws_eff, x0, ref)
        for i in range(DEPTH): g_acc[i] += gs[i] / N_CAL
    t_step += 1
    for i in range(DEPTH):
        m[i] = b1*m[i] + (1-b1)*g_acc[i]
        v[i] = b2*v[i] + (1-b2)*g_acc[i]**2
        tilts[i] -= lr * (m[i]/(1-b1**t_step)) / (np.sqrt(v[i]/(1-b2**t_step)) + eps)
        tilts[i] = np.clip(tilts[i], -half_step[i], half_step[i])  # inches, not yards

ws_grad = best[1]
print(f"Gradient pre-tilt (clipped, best checkpoint): test {eval_test(ws_grad):.5f} | flips {flips(ws_grad)}/50")

# Longer random search (the v1 approach, given 5x the budget), same clipping
rng2 = np.random.default_rng(7)
tilts_r = [np.zeros_like(W) for W in layers]
best_ce = calib_err(q_naive)
sigma = 0.02
for it in range(2000):
    li = rng2.integers(0, DEPTH)
    prop = np.clip(tilts_r[li] + rng2.normal(0, sigma, layers[li].shape),
                   -half_step[li], half_step[li])
    trial = [quantize(layers[i] + (prop if i == li else tilts_r[i])) for i in range(DEPTH)]
    ce = calib_err(trial)
    if ce < best_ce:
        best_ce = ce; tilts_r[li] = prop
ws_rand = [quantize(layers[i] + tilts_r[i]) for i in range(DEPTH)]
print(f"Random search x2000 (clipped): test {eval_test(ws_rand):.5f} | flips {flips(ws_rand)}/50")

# Combo: start from random-search result, polish with gentle gradient
tilts_c = [t.copy() for t in tilts_r]
m = [np.zeros_like(W) for W in layers]; v = [np.zeros_like(W) for W in layers]
best_c = (calib_err(ws_rand), [W.copy() for W in ws_rand])
t_step = 0
for epoch in range(100):
    ws_eff = [quantize(layers[i] + tilts_c[i]) for i in range(DEPTH)]
    ce = calib_err(ws_eff)
    if ce < best_c[0]:
        best_c = (ce, [W.copy() for W in ws_eff])
    g_acc = [np.zeros_like(W) for W in layers]
    for x0, ref in zip(x0s, refs):
        gs = gradients(ws_eff, x0, ref)
        for i in range(DEPTH): g_acc[i] += gs[i] / N_CAL
    t_step += 1
    for i in range(DEPTH):
        m[i] = b1*m[i] + (1-b1)*g_acc[i]
        v[i] = b2*v[i] + (1-b2)*g_acc[i]**2
        tilts_c[i] -= lr * (m[i]/(1-b1**t_step)) / (np.sqrt(v[i]/(1-b2**t_step)) + eps)
        tilts_c[i] = np.clip(tilts_c[i], -half_step[i], half_step[i])
ws_combo = best_c[1]
print(f"Combo (random then gradient polish): test {eval_test(ws_combo):.5f} | flips {flips(ws_combo)}/50")

nt = eval_test(q_naive)
for name, ws in [("gradient", ws_grad), ("random x2000", ws_rand), ("combo", ws_combo)]:
    e = eval_test(ws)
    print(f"  {name}: {100*(1-e/nt):.1f}% error removed vs naive")
