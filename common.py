"""Shared utilities for EN4554 Assignment 2 (CNNs on CIFAR-10).

Data are loaded once, normalised with TRAINING-split statistics, and kept as
tensors on the GPU. Augmentations are done on the GPU in batches, so training
is fast and there is no DataLoader overhead.
"""
import os, json, random, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                     # no display on a server
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SEED = 0                            # the 45k/5k split never changes


# ----------------------------------------------------------------------------
# Arguments, seeds, output folders
# ----------------------------------------------------------------------------
def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default="./data", help="CIFAR-10 download folder")
    p.add_argument("--out", default="./results", help="results folder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny data + 1-2 epochs")
    p.add_argument("--fake", action="store_true",
                   help="random data instead of CIFAR-10 (offline testing only)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision")
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def out_dir(args, name):
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# >>> REPORT: data
def load_cifar10(args, load_test=False, device=DEVICE):
    """45k train / 5k val split of the CIFAR-10 training set, normalised per
    channel with the statistics of the 45k training images only."""
    if args.fake:
        rng = np.random.RandomState(123)
        xtr = rng.randint(0, 256, (50000, 32, 32, 3), dtype=np.uint8)
        ytr = rng.randint(0, 10, 50000)
        xte = rng.randint(0, 256, (10000, 32, 32, 3), dtype=np.uint8)
        yte = rng.randint(0, 10, 10000)
    else:
        from torchvision.datasets import CIFAR10
        tr = CIFAR10(args.data, train=True, download=True)
        xtr, ytr = tr.data, np.array(tr.targets)
        if load_test:                       # test set is touched ONLY in Task 4
            te = CIFAR10(args.data, train=False, download=True)
            xte, yte = te.data, np.array(te.targets)

    perm = np.random.RandomState(SPLIT_SEED).permutation(50000)
    tr_idx, va_idx = perm[:45000], perm[45000:]

    def to_tensor(a):                       # uint8 NHWC -> float NCHW in [0,1]
        return torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255)

    x = to_tensor(xtr)
    mean = x[tr_idx].mean(dim=(0, 2, 3))    # training statistics only
    std = x[tr_idx].std(dim=(0, 2, 3))

    def norm(t):
        return ((t - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)).to(device)

    y = torch.from_numpy(ytr).long()
    d = dict(xtr=norm(x[tr_idx]), ytr=y[tr_idx].to(device),
             xva=norm(x[va_idx]), yva=y[va_idx].to(device),
             mean=mean.tolist(), std=std.tolist())
    if load_test:
        d["xte"] = norm(to_tensor(xte))
        d["yte"] = torch.from_numpy(yte).long().to(device)
    if args.quick:                          # shrink everything for a smoke test
        d["xtr"], d["ytr"] = d["xtr"][:2000], d["ytr"][:2000]
        d["xva"], d["yva"] = d["xva"][:1000], d["yva"][:1000]
        if load_test:
            d["xte"], d["yte"] = d["xte"][:1000], d["yte"][:1000]
    return d
# <<< REPORT


def batches(x, y, bs, shuffle=True):
    n = len(x)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield x[b], y[b]


# ----------------------------------------------------------------------------
# Augmentations (all implemented by hand, on GPU batches)
# ----------------------------------------------------------------------------
# >>> REPORT: augment
def random_flip(x):
    """Horizontal flip of each image with probability 0.5."""
    mask = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(mask[:, None, None, None], x.flip(3), x)


def random_crop(x, pad=4):
    """Zero-pad by `pad` pixels and take a random 32x32 crop per image."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = i[:, None] + torch.arange(H, device=x.device)[None]      # B x H
    cols = j[:, None] + torch.arange(W, device=x.device)[None]      # B x W
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    return xp[b, c, rows[:, None, :, None], cols[:, None, None, :]]


def cutout(x, size=16):
    """Cutout (DeVries & Taylor 2017): zero a random size x size square per
    image. The centre can be near the border, so the square may be clipped."""
    B, C, H, W = x.shape
    cy = torch.randint(0, H, (B, 1, 1), device=x.device)
    cx = torch.randint(0, W, (B, 1, 1), device=x.device)
    ys = torch.arange(H, device=x.device)[None, :, None]
    xs = torch.arange(W, device=x.device)[None, None, :]
    h = size // 2
    inside = (ys >= cy - h) & (ys < cy + h) & (xs >= cx - h) & (xs < cx + h)
    return x * (~inside)[:, None].to(x.dtype)


def mixup(x, y, alpha):
    """Mixup (Zhang et al. 2018). Returns mixed inputs, both label sets and
    lambda; the loss is lam*CE(out, y_a) + (1-lam)*CE(out, y_b)."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam
# <<< REPORT


def make_aug(names, cutout_size=16):
    """names: list from {'flip','crop','cutout'} -> function x -> x."""
    names = list(names or [])
    if not names:
        return None

    def aug(x):
        if "flip" in names:
            x = random_flip(x)
        if "crop" in names:
            x = random_crop(x)
        if "cutout" in names:
            x = cutout(x, cutout_size)
        return x
    return aug


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def autocast(enabled):
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                          enabled=enabled and DEVICE.type == "cuda")


@torch.no_grad()
def evaluate(model, x, y, bs=1000, amp=True):
    model.eval()
    tot_loss, correct = 0.0, 0
    for i in range(0, len(x), bs):
        with autocast(amp):
            out = model(x[i:i + bs])
        out = out.float()
        tot_loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return tot_loss / len(x), correct / len(x)


# >>> REPORT: train
def train(model, data, epochs, optimizer, scheduler=None, bs=128, aug=None,
          mixup_alpha=0.0, label_smoothing=0.0, amp=True, xtr=None, ytr=None,
          clean_train_eval=False, tag=""):
    """Generic training loop. `scheduler` is stepped every iteration.
    Returns a history dict with per-iteration and per-epoch curves."""
    xtr = data["xtr"] if xtr is None else xtr
    ytr = data["ytr"] if ytr is None else ytr
    use_amp = amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    H = {k: [] for k in ["iter_loss", "train_loss", "train_acc", "val_loss",
                         "val_acc", "train_acc_clean", "train_loss_clean", "lr"]}
    H["diverged"] = False
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        losses, correct, seen = [], torch.zeros((), device=DEVICE), 0
        for xb, yb in batches(xtr, ytr, bs):
            if aug is not None:
                xb = aug(xb)
            with autocast(use_amp):
                if mixup_alpha > 0:
                    xb, ya, yb2, lam = mixup(xb, yb, mixup_alpha)
                    out = model(xb)
                    loss = (lam * F.cross_entropy(out, ya, label_smoothing=label_smoothing)
                            + (1 - lam) * F.cross_entropy(out, yb2, label_smoothing=label_smoothing))
                else:
                    out = model(xb)
                    loss = F.cross_entropy(out, yb, label_smoothing=label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.detach().float())
            correct += (out.argmax(1) == yb).sum()
            seen += len(yb)
        ep_losses = torch.stack(losses).cpu().tolist()
        H["iter_loss"] += ep_losses
        H["train_loss"].append(float(np.mean(ep_losses)))
        H["train_acc"].append(correct.item() / seen)
        H["lr"].append(optimizer.param_groups[0]["lr"])
        vl, va = evaluate(model, data["xva"], data["yva"], amp=amp)
        H["val_loss"].append(vl)
        H["val_acc"].append(va)
        if clean_train_eval:                # accuracy on un-augmented train set
            tl, ta = evaluate(model, xtr, ytr, amp=amp)
            H["train_loss_clean"].append(tl)
            H["train_acc_clean"].append(ta)
        print(f"[{tag}] ep {ep + 1:3d}/{epochs} loss {H['train_loss'][-1]:.4f} "
              f"train_acc {H['train_acc'][-1]:.4f} val_acc {va:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if not math.isfinite(H["train_loss"][-1]):
            print(f"[{tag}] diverged (non-finite loss); stopping this run.")
            H["diverged"] = True
            break
    return H
# <<< REPORT


def smooth(v, k=50):
    v = np.asarray(v, dtype=float)
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def epochs_for(args, full):
    return 2 if args.quick else full
