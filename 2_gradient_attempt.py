import numpy as np

rng = np.random.default_rng(42)

DEPTH, DIM, STEPS, BITS = 12, 64, 30, 4
layers = [rng.normal(0, 1/np.sqrt(DIM), (DIM, DIM)) for _ in range(DEPTH)]

def quantize(W, bits=BITS):
    scale = np.abs(W).max(axis=1, keepdims=True) / (2**(bits-1) - 1)
    return np.round(W / scale) * scale

def reason(ws, x0, steps=STEPS):
    x = x0.copy(); traj = []
    for _ in range(steps):
        for W in ws:
            x = np.tanh(W @ x)
        traj.append(x.copy())
    return np.array(traj)

# Calibration + reference trajectories (the "answer key")
N_CAL = 8
x0s = [rng.normal(0, 1, DIM) for _ in range(N_CAL)]
refs = [reason(layers, x0) for x0 in x0s]

# ---------- Forward pass that remembers everything (needed to learn) ----------
def forward_record(ws, x0):
    x = x0.copy()
    xs_in = []   # input to each (step, layer)
    zs = []      # pre-activation of each (step, layer)
    traj = []
    for t in range(STEPS):
        for l in range(DEPTH):
            xs_in.append(x)
            z = ws[l] @ x
            zs.append(z)
            x = np.tanh(z)
        traj.append(x.copy())
    return np.array(traj), xs_in, zs

# ---------- Backward pass: which way should each weight move? ----------
def gradients(ws, x0, ref):
    traj, xs_in, zs = forward_record(ws, x0)
    grads = [np.zeros_like(W) for W in ws]
    dx = np.zeros(DIM)  # gradient flowing backward through the state
    loss = 0.0
    idx = STEPS * DEPTH - 1
    for t in reversed(range(STEPS)):
        diff = traj[t] - ref[t]
        loss += np.mean(diff**2)
        dx = dx + (2.0 / (DIM * STEPS)) * diff
        for l in reversed(range(DEPTH)):
            z = zs[idx]; x_in = xs_in[idx]
            dz = dx * (1 - np.tanh(z)**2)
            grads[l] += np.outer(dz, x_in)
            dx = ws[l].T @ dz
            idx -= 1
    return loss / STEPS, grads

# ---------- Straight-through training of the pre-tilt ----------
# Effective weights = quantize(original + tilt). We pretend the rounding is
# transparent when computing directions (the standard "straight-through" trick),
# so the tilt learns to pre-cancel the rounding damage across the WHOLE chain.
tilts = [np.zeros_like(W) for W in layers]
m = [np.zeros_like(W) for W in layers]  # momentum (Adam)
v = [np.zeros_like(W) for W in layers]
lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8

def eval_test(ws, test_x0s):
    e = 0.0
    for x0 in test_x0s:
        ref = reason(layers, x0)
        e += np.mean((reason(ws, x0) - ref)**2)
    return e / len(test_x0s)

test_x0s = [rng.normal(0, 1, DIM) for _ in range(20)]
readout = rng.normal(0, 1, DIM)
def flips(ws):
    f = 0
    for x0 in test_x0s:
        a = reason(layers, x0)[-1] @ readout
        b = reason(ws, x0)[-1] @ readout
        f += (np.sign(a) != np.sign(b))
    return f

q_naive = [quantize(W) for W in layers]
print(f"Naive 4-bit        | test error {eval_test(q_naive, test_x0s):.5f} | decision flips {flips(q_naive)}/20")

t_step = 0
for epoch in range(60):
    total = 0.0
    g_acc = [np.zeros_like(W) for W in layers]
    ws_eff = [quantize(layers[i] + tilts[i]) for i in range(DEPTH)]
    for x0, ref in zip(x0s, refs):
        loss, gs = gradients(ws_eff, x0, ref)
        total += loss
        for i in range(DEPTH):
            g_acc[i] += gs[i] / N_CAL
    t_step += 1
    for i in range(DEPTH):
        m[i] = b1*m[i] + (1-b1)*g_acc[i]
        v[i] = b2*v[i] + (1-b2)*g_acc[i]**2
        mh = m[i] / (1 - b1**t_step)
        vh = v[i] / (1 - b2**t_step)
        tilts[i] -= lr * mh / (np.sqrt(vh) + eps)
    if epoch % 15 == 14 or epoch == 0:
        ws_now = [quantize(layers[i] + tilts[i]) for i in range(DEPTH)]
        print(f"  after {epoch+1:2d} rounds  | calib {total/N_CAL:.5f} | test {eval_test(ws_now, test_x0s):.5f} | flips {flips(ws_now)}/20")

ws_final = [quantize(layers[i] + tilts[i]) for i in range(DEPTH)]
final_test = eval_test(ws_final, test_x0s)
naive_test = eval_test(q_naive, test_x0s)
print()
print(f"FINAL: naive {naive_test:.5f} -> smart pre-tilt {final_test:.5f}  ({100*(1-final_test/naive_test):.1f}% error removed)")
print(f"Decision flips: naive {flips(q_naive)}/20 -> smart pre-tilt {flips(ws_final)}/20")

# Sanity check vs a stronger baseline: same smart search allowed to tune
# each layer ONLY for its own output (like today's GPTQ-style methods),
# not for the whole chain. Does whole-chain awareness actually matter?
def per_layer_calibrated():
    out = []
    for i, W in enumerate(layers):
        # collect typical inputs to this layer from calibration runs
        ins = []
        for x0 in x0s:
            x = x0.copy()
            for t in range(STEPS):
                for l in range(DEPTH):
                    if l == i: ins.append(x.copy())
                    x = np.tanh((layers[l] if l != i else W) @ x)
        X = np.array(ins).T  # DIM x samples
        # find tilt minimizing ||(quant(W+T) - W) X|| via simple gradient steps (STE)
        T = np.zeros_like(W); mm = np.zeros_like(W); vv = np.zeros_like(W)
        for s in range(1, 61):
            Q = quantize(W + T)
            E = (Q - W) @ X
            g = (2.0 / X.shape[1]) * E @ X.T
            mm = b1*mm + (1-b1)*g; vv = b2*vv + (1-b2)*g**2
            T -= lr * (mm/(1-b1**s)) / (np.sqrt(vv/(1-b2**s)) + eps)
        out.append(quantize(W + T))
    return out

ws_perlayer = per_layer_calibrated()
print(f"Per-layer-only calibration (today's style): test {eval_test(ws_perlayer, test_x0s):.5f} | flips {flips(ws_perlayer)}/20")
