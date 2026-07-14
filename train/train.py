"""Quantization-aware training of ternary-weight MNIST classifiers on 8x8
binarized inputs, for on-chip inference (Tiny Tapeout, weights as constants).

Variants:
  linear: 64 -> 10, ternary weights {-1,0,+1}, integer bias
  mlp:    64 -> 16 -> 10, ternary weights, binary {0,1} hidden activations,
          integer biases (hidden binarization threshold folded into bias)

Exports (per variant):
  weights/<variant>.json   integer weights + biases
  test/vectors_<variant>.json   test images as 64-bit hex + integer-model
                                predictions (ground truth for RTL testbench)

The integer reference model in this file is the exact twin of the RTL:
  L1:  acc[j] = bias1[j] + sum_i x[i]*W1[j][i]     (x in {0,1})
  MLP: h[j]   = 1 if acc[j] >= 0 else 0
       acc2[c]= bias2[c] + sum_j h[j]*W2[c][j]
  out: argmax, first (lowest index) wins ties
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import pool_8x8  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
N_VECTORS = 200  # test vectors exported for the cocotb testbench

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_mnist_pooled():
    """Return (train8, train_y, test8, test_y) with 8x8 float-pooled images."""
    from torchvision import datasets

    data_dir = os.path.join(ROOT, "data")
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)

    def pool_all(ds):
        imgs = ds.data.numpy().astype(np.float64) / 255.0  # (N,28,28)
        x = torch.from_numpy(imgs).unsqueeze(1).float()
        pooled = F.adaptive_avg_pool2d(x, 8).squeeze(1).numpy()  # (N,8,8)
        return pooled, ds.targets.numpy()

    tr_x, tr_y = pool_all(train)
    te_x, te_y = pool_all(test)

    # Sanity: torch adaptive pool must match our from-scratch pool_8x8
    # (the RP2040 demo and cocotb TB use the from-scratch version).
    ref = pool_8x8(train.data.numpy()[0].astype(np.float64) / 255.0)
    assert np.allclose(ref, tr_x[0], atol=1e-6), "pooling mismatch"
    return tr_x, tr_y, te_x, te_y


# --------------------------------------------------------------------------
# Ternary QAT building blocks
# --------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """Linear layer whose weights are ternarized (TWN-style) in the forward
    pass with a straight-through estimator."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.frozen_delta = None  # freeze after warm start to stop mask thrash

    def freeze_delta(self):
        with torch.no_grad():
            self.frozen_delta = float(0.7 * self.weight.abs().mean())

    def ternarize(self):
        w = self.weight
        delta = self.frozen_delta if self.frozen_delta is not None \
            else 0.7 * w.abs().mean()
        mask = (w.abs() > delta).float()
        wq = torch.sign(w) * mask
        n = mask.sum().clamp(min=1)
        alpha = (w.abs() * mask).sum() / n  # per-tensor scale
        return wq, alpha

    def forward(self, x):
        wq, alpha = self.ternarize()
        w_eff = alpha * wq
        # STE: gradients flow to the float master weights / float bias
        w = self.weight + (w_eff - self.weight).detach()
        b_int = alpha * torch.round(self.bias / alpha)
        b = self.bias + (b_int - self.bias).detach()
        return F.linear(x, w, b)

    def export_int(self):
        """Integer weights {-1,0,1} and integer bias (bias / alpha)."""
        with torch.no_grad():
            wq, alpha = self.ternarize()
            b_int = torch.round(self.bias / alpha)
            return (wq.numpy().astype(int),
                    b_int.numpy().astype(int),
                    float(alpha))


class BinaryAct(nn.Module):
    """Hard {0,1} activation with straight-through gradient (sigmoid
    surrogate — a clipped-identity window leaves most units without
    gradient because pre-activations span several units of scale)."""

    def forward(self, x):
        hard = (x >= 0).float()
        soft = torch.sigmoid(2.0 * x)
        return soft + (hard - soft).detach()


class LinearModel(nn.Module):
    def __init__(self, ternary=True):
        super().__init__()
        self.fc = TernaryLinear(64, 10) if ternary else nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x)


class MLPModel(nn.Module):
    def __init__(self, hidden=16, ternary=True, binact=True):
        super().__init__()
        L = TernaryLinear if ternary else nn.Linear
        self.fc1 = L(64, hidden)
        self.act = BinaryAct() if binact else nn.ReLU()
        self.fc2 = L(hidden, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def train_model(model, xb, yb, epochs=20, lr=1e-3, batch=256,
                select_fn=None, log_every=0):
    """Train; if select_fn is given (model -> score), keep and restore the
    best-scoring epoch's parameters (QAT with hard ternarization oscillates,
    so last-epoch weights are often far from the best)."""
    import copy

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xb_t = torch.from_numpy(xb).float()
    yb_t = torch.from_numpy(yb).long()
    n = len(xb_t)
    best_score, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_t[idx]), yb_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        if select_fn is not None:
            score = select_fn(model)
            if score > best_score:
                best_score, best_state = score, copy.deepcopy(model.state_dict())
            if log_every and (ep + 1) % log_every == 0:
                print(f"    epoch {ep + 1:3d}  val score {score:.4f} "
                      f"(best {best_score:.4f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_model(model, x, y):
    model.eval()
    logits = model(torch.from_numpy(x).float())
    return (logits.argmax(1).numpy() == y).mean()


# --------------------------------------------------------------------------
# Exact integer twin of the RTL
# --------------------------------------------------------------------------

def int_forward(weights, x_bits):
    """x_bits: (N,64) in {0,1}. Returns predictions using pure int math."""
    w1 = np.array(weights["w1"])          # (H,64) or (10,64)
    b1 = np.array(weights["b1"])
    acc1 = x_bits @ w1.T + b1             # (N,H)
    if "w2" in weights:
        h = (acc1 >= 0).astype(int)
        w2 = np.array(weights["w2"])      # (10,H)
        b2 = np.array(weights["b2"])
        acc = h @ w2.T + b2
    else:
        acc = acc1
    return np.argmax(acc, axis=1)          # np.argmax = first max wins ties


def check_acc_ranges(weights):
    """Assert every accumulator fits a signed 8-bit register."""
    w1 = np.array(weights["w1"]); b1 = np.array(weights["b1"])
    hi = b1 + np.clip(w1, 0, None).sum(1)
    lo = b1 + np.clip(w1, None, 0).sum(1)
    assert hi.max() <= 127 and lo.min() >= -128, f"L1 acc overflow {lo.min()}..{hi.max()}"
    if "w2" in weights:
        w2 = np.array(weights["w2"]); b2 = np.array(weights["b2"])
        hi2 = b2 + np.clip(w2, 0, None).sum(1)
        lo2 = b2 + np.clip(w2, None, 0).sum(1)
        assert hi2.max() <= 127 and lo2.min() >= -128, "L2 acc overflow"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def export_linear(model, thresh):
    w, b, alpha = model.fc.export_int()
    b = np.clip(b, -63, 63)  # keep L1 accumulator within int8
    return {"arch": "linear", "w1": w.tolist(), "b1": b.tolist(),
            "alpha1": alpha, "bin_threshold": thresh}


def export_mlp(model, thresh):
    w1, b1, a1 = model.fc1.export_int()
    w2, b2, a2 = model.fc2.export_int()
    h = len(w1)
    return {"arch": "mlp", "hidden": h,
            "w1": w1.tolist(), "b1": np.clip(b1, -63, 63).tolist(),
            "alpha1": a1,
            "w2": w2.tolist(), "b2": np.clip(b2, -(127 - h), 127 - h).tolist(),
            "alpha2": a2, "bin_threshold": thresh}


def main():
    print("Loading MNIST (downloads ~12 MB on first run)...")
    tr_x, tr_y_all, te_x, te_y = load_mnist_pooled()

    # 55k train / 5k validation split (validation drives threshold choice
    # and best-epoch selection; the 10k test set is only used for reporting)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_y_all))
    tr_idx, va_idx = perm[:55000], perm[55000:]

    def split(thresh):
        xb_all = (tr_x > thresh).reshape(-1, 64).astype(np.float32)
        xt = (te_x > thresh).reshape(-1, 64).astype(np.float32)
        return (xb_all[tr_idx], tr_y_all[tr_idx],
                xb_all[va_idx], tr_y_all[va_idx], xt)

    def int_val_acc(export_fn, xv_bits, yv, thresh):
        """Selection score: exact integer-model accuracy on validation."""
        def score(model):
            return (int_forward(export_fn(model, thresh), xv_bits) == yv).mean()
        return score

    # ---- 1. Binarization threshold sweep (short ternary-linear QAT) ----
    print("\n== Binarization threshold sweep (6-epoch ternary linear) ==")
    best_thresh, best_acc = None, -1
    for thresh in (0.10, 0.15, 0.20, 0.25):
        torch.manual_seed(SEED)
        xb, yb, xv, yv, _ = split(thresh)
        xv_bits = xv.astype(int)
        m = train_model(LinearModel(ternary=True), xb, yb, epochs=6,
                        select_fn=int_val_acc(export_linear, xv_bits, yv, thresh))
        acc = (int_forward(export_linear(m, thresh), xv_bits) == yv).mean()
        print(f"  threshold {thresh:.2f}: int-model val acc {acc:.4f}")
        if acc > best_acc:
            best_thresh, best_acc = thresh, acc
    print(f"  -> selected threshold {best_thresh}")

    xb, yb, xv, yv, xt = split(best_thresh)
    xv_bits, xt_bits = xv.astype(int), xt.astype(int)
    results = {}

    # ---- 2. Float baselines (upper bounds, not exported) ----
    print("\n== Float baselines ==")
    torch.manual_seed(SEED)
    flin = train_model(LinearModel(ternary=False), xb, yb, epochs=30,
                       select_fn=lambda m: eval_model(m, xv, yv))
    results["float_linear"] = eval_model(flin, xt, te_y)
    print(f"  float linear 64->10:          {results['float_linear']:.4f}")

    # ---- 3. Ternary QAT (warm-started from the float weights) ----
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)

    print("\n== Ternary linear 64->10 (QAT) ==")
    torch.manual_seed(SEED)
    lin = LinearModel(ternary=True)
    lin.fc.weight.data = flin.fc.weight.data.clone()
    lin.fc.bias.data = flin.fc.bias.data.clone()
    lin.fc.freeze_delta()
    lin = train_model(lin, xb, yb, epochs=40, lr=3e-4, log_every=10,
                      select_fn=int_val_acc(export_linear, xv_bits, yv,
                                            best_thresh))
    weights_lin = export_linear(lin, best_thresh)
    check_acc_ranges(weights_lin)
    results["int_linear"] = (int_forward(weights_lin, xt_bits) == te_y).mean()
    print(f"  integer-model TEST acc {results['int_linear']:.4f}")

    exported = {"linear": weights_lin}
    for H in (16, 32):
        print(f"\n== Ternary MLP 64->{H}->10 (QAT, binary hidden) ==")
        # float ReLU -> float binary-activation -> ternary QAT
        torch.manual_seed(SEED)
        frelu = train_model(MLPModel(hidden=H, ternary=False, binact=False),
                            xb, yb, epochs=30,
                            select_fn=lambda m: eval_model(m, xv, yv))
        results[f"float_mlp{H}_relu"] = eval_model(frelu, xt, te_y)
        torch.manual_seed(SEED)
        fbin = MLPModel(hidden=H, ternary=False, binact=True)
        fbin.load_state_dict(frelu.state_dict())
        fbin = train_model(fbin, xb, yb, epochs=30,
                           select_fn=lambda m: eval_model(m, xv, yv))
        results[f"float_mlp{H}_binact"] = eval_model(fbin, xt, te_y)

        mlp = MLPModel(hidden=H, ternary=True)
        mlp.fc1.weight.data = fbin.fc1.weight.data.clone()
        mlp.fc1.bias.data = fbin.fc1.bias.data.clone()
        mlp.fc2.weight.data = fbin.fc2.weight.data.clone()
        mlp.fc2.bias.data = fbin.fc2.bias.data.clone()
        mlp.fc1.freeze_delta()
        mlp.fc2.freeze_delta()
        mlp = train_model(mlp, xb, yb, epochs=60, lr=3e-4, log_every=20,
                          select_fn=int_val_acc(export_mlp, xv_bits, yv,
                                                best_thresh))
        wts = export_mlp(mlp, best_thresh)
        check_acc_ranges(wts)
        exported[f"mlp{H}"] = wts
        results[f"int_mlp{H}"] = (int_forward(wts, xt_bits) == te_y).mean()
        print(f"  integer-model TEST acc {results[f'int_mlp{H}']:.4f}")

    # ---- 4. Export weights + test vectors ----
    for name, wts in exported.items():
        with open(os.path.join(ROOT, "weights", f"{name}.json"), "w") as f:
            json.dump(wts, f)

        rng = np.random.RandomState(SEED)
        sel = rng.choice(len(te_y), N_VECTORS, replace=False)
        words = []
        for i in sel:
            bits = xt_bits[i]
            word = 0
            for k in range(64):
                if bits[k]:
                    word |= 1 << k
            words.append(word)
        preds = int_forward(wts, xt_bits[sel])
        vectors = {
            "variant": name,
            "bin_threshold": best_thresh,
            "images": [f"{w:016x}" for w in words],
            "expected": preds.tolist(),
            "labels": te_y[sel].tolist(),
        }
        with open(os.path.join(ROOT, "test", f"vectors_{name}.json"), "w") as f:
            json.dump(vectors, f)

    with open(os.path.join(ROOT, "weights", "results.json"), "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    print("\n== Summary (test accuracy, 10000 images) ==")
    for k, v in results.items():
        print(f"  {k:15s} {v:.4f}")
    print(f"\nExported: {', '.join('weights/' + n + '.json' for n in exported)}"
          f" + test/vectors_*.json ({N_VECTORS} images each)")
    print(f"Binarization threshold: {best_thresh} "
          f"(update BIN_THRESHOLD in train/preprocess.py if != 0.25)")


if __name__ == "__main__":
    main()
