"""Task 1: backpropagation by hand (no autograd in the model itself).
(a) Linear / ReLU / Sigmoid / softmax-cross-entropy with explicit forward and
    backward, assembled into an MLP.
(b) Gradient check against autograd and centred finite differences.
(c) SGD training on flattened CIFAR-10.
(d) Per-layer gradient norms of a 10-layer MLP at init: sigmoid vs ReLU.
"""
from common import *


# >>> REPORT: layers
class Linear:
    def __init__(self, din, dout, init="kaiming", dtype=torch.float32, device=DEVICE):
        std = {"kaiming": math.sqrt(2.0 / din),          # He, for ReLU
               "xavier": math.sqrt(2.0 / (din + dout)),  # Glorot, for sigmoid/tanh
               "lecun": math.sqrt(1.0 / din)}[init]
        self.W = torch.randn(din, dout, dtype=dtype, device=device) * std
        self.b = torch.zeros(dout, dtype=dtype, device=device)
        self.dW, self.db = torch.zeros_like(self.W), torch.zeros_like(self.b)

    def forward(self, x):                 # y = x W + b
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):                # g = dL/dy  (N x dout)
        self.dW = self.x.T @ g            # dL/dW = x^T g
        self.db = g.sum(0)                # dL/db = sum over batch
        return g @ self.W.T               # dL/dx = g W^T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU:
    def forward(self, x):
        self.mask = (x > 0).to(x.dtype)
        return x * self.mask

    def backward(self, g):                # derivative is 1 where x > 0
        return g * self.mask

    def params(self):
        return []

    def grads(self):
        return []


class Sigmoid:
    def forward(self, x):
        self.s = 1.0 / (1.0 + torch.exp(-x))
        return self.s

    def backward(self, g):                # sigma' = sigma (1 - sigma) <= 1/4
        return g * self.s * (1 - self.s)

    def params(self):
        return []

    def grads(self):
        return []


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        z = logits - logits.max(1, keepdim=True).values   # numerical stability
        logp = z - z.exp().sum(1, keepdim=True).log()       # log-softmax
        self.p, self.y = logp.exp(), y
        return -logp[torch.arange(len(y)), y].mean()

    def backward(self):                   # dL/dz = (softmax - onehot) / N
        g = self.p.clone()
        g[torch.arange(len(self.y)), self.y] -= 1
        return g / len(self.y)


class MLP:
    def __init__(self, sizes, act="relu", init=None, dtype=torch.float32, device=DEVICE):
        init = init or ("kaiming" if act == "relu" else "xavier")
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b, init, dtype, device))
            if i < len(sizes) - 2:
                self.layers.append(ReLU() if act == "relu" else Sigmoid())
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.criterion.forward(self.forward(x), y)

    def backward(self):
        g = self.criterion.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def linears(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def grads(self):
        return [g for l in self.layers for g in l.grads()]

    def sgd_step(self, lr):
        for p, g in zip(self.params(), self.grads()):
            p.sub_(lr * g)
# <<< REPORT


# >>> REPORT: gradcheck
def autograd_grads(mlp, x, y):
    """Same network rebuilt with torch ops + autograd, sharing the weights."""
    ps = [p.detach().clone().requires_grad_(True) for p in mlp.params()]
    h, k = x, 0
    for layer in mlp.layers:
        if isinstance(layer, Linear):
            h = h @ ps[k] + ps[k + 1]
            k += 2
        elif isinstance(layer, ReLU):
            h = torch.relu(h)
        else:
            h = torch.sigmoid(h)
    loss = F.cross_entropy(h, y)
    loss.backward()
    return [p.grad for p in ps]


def finite_diff(mlp, x, y, p, idx, eps=1e-6):
    """Centred difference (L(p+eps) - L(p-eps)) / 2eps at flat indices idx."""
    flat, out = p.view(-1), []
    for i in idx:
        old = flat[i].item()
        flat[i] = old + eps
        lp = mlp.loss(x, y).item()
        flat[i] = old - eps
        lm = mlp.loss(x, y).item()
        flat[i] = old
        out.append((lp - lm) / (2 * eps))
    return torch.tensor(out, dtype=p.dtype, device=p.device)


def rel_err(a, b):
    return ((a - b).norm() / (a.norm() + b.norm() + 1e-30)).item()
# <<< REPORT


def gradient_check(mlp, x, y, n_fd=None, name=""):
    mlp.loss(x, y)
    mlp.backward()
    manual = [g.clone() for g in mlp.grads()]
    auto = autograd_grads(mlp, x, y)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for k, (p, gm, ga) in enumerate(zip(mlp.params(), manual, auto)):
        n = p.numel()
        idx = torch.arange(n) if (n_fd is None or n <= n_fd) else torch.randperm(n, generator=gen)[:n_fd]
        fd = finite_diff(mlp, x, y, p, idx.tolist())
        gm_s = gm.reshape(-1)[idx.to(gm.device)]
        rows.append(dict(
            model=name, param=f"{'W' if k % 2 == 0 else 'b'}{k // 2 + 1}", shape=list(p.shape),
            rel_err_autograd=rel_err(gm, ga),
            allclose_autograd=bool(torch.allclose(gm, ga, rtol=1e-6, atol=1e-10)),
            n_fd_entries=len(idx), rel_err_fd=rel_err(gm_s, fd),
            allclose_fd=bool(torch.allclose(gm_s, fd, rtol=1e-4, atol=1e-7))))
    return rows


def main():
    args = base_parser("Task 1").parse_args()
    set_seed(args.seed)
    od = out_dir(args, "task1")
    data = load_cifar10(args)
    Xtr, Xva = data["xtr"].flatten(1), data["xva"].flatten(1)
    ytr, yva = data["ytr"], data["yva"]

    # ---------------- (b) gradient check (float64) ----------------
    print("== Task 1(b): gradient check ==")
    rows = []
    torch.manual_seed(0)
    xs = torch.randn(8, 20, dtype=torch.float64, device=DEVICE)
    ys = torch.randint(0, 10, (8,), device=DEVICE)
    for act in ["relu", "sigmoid"]:
        tiny = MLP([20, 16, 12, 10], act=act, dtype=torch.float64)
        rows += gradient_check(tiny, xs, ys, n_fd=None, name=f"tiny-{act} (all entries)")
    xb, yb = Xtr[:16].double(), ytr[:16]
    big = MLP([3072, 128, 64, 10], act="relu", dtype=torch.float64)
    rows += gradient_check(big, xb, yb, n_fd=40, name="cifar-relu (40 random entries/tensor for FD)")
    save_json(rows, os.path.join(od, "gradcheck.json"))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "gradcheck.csv"), index=False)
    print(df.to_string(index=False))

    # ---------------- (c) train with SGD ----------------
    print("== Task 1(c): SGD training ==")
    set_seed(args.seed)
    epochs, lr, bs = epochs_for(args, 30), 0.02, 128
    mlp = MLP([3072, 512, 256, 10], act="relu")
    H = dict(iter_loss=[], train_loss=[], val_loss=[], val_acc=[])
    for ep in range(epochs):
        ls = []
        for x, y in batches(Xtr, ytr, bs):
            loss = mlp.loss(x, y)
            mlp.backward()
            mlp.sgd_step(lr)
            ls.append(loss.item())
        with torch.no_grad():
            logits = mlp.forward(Xva)
            vl = mlp.criterion.forward(logits, yva).item()
            va = (logits.argmax(1) == yva).float().mean().item()
        H["iter_loss"] += ls
        H["train_loss"].append(float(np.mean(ls)))
        H["val_loss"].append(vl)
        H["val_acc"].append(va)
        print(f"ep {ep + 1:2d} train_loss {H['train_loss'][-1]:.4f} val_loss {vl:.4f} val_acc {va:.4f}", flush=True)
    save_json(H, os.path.join(od, "mlp_sgd_history.json"))
    save_json(dict(final_val_acc=H["val_acc"][-1], best_val_acc=max(H["val_acc"]),
                   epochs=epochs, lr=lr, batch_size=bs, arch=[3072, 512, 256, 10]),
              os.path.join(od, "mlp_sgd_summary.json"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(H["iter_loss"], alpha=0.3, label="mini-batch loss")
    ax[0].plot(np.arange(len(smooth(H["iter_loss"]))) + 49, smooth(H["iter_loss"]), label="moving avg (50)")
    ax[0].set(xlabel="iteration", ylabel="cross-entropy", title="MLP training loss (manual backprop, SGD)")
    ax[0].legend()
    e = np.arange(1, len(H["val_acc"]) + 1)
    ax[1].plot(e, H["train_loss"], "o-", label="train loss")
    ax[1].plot(e, H["val_loss"], "s-", label="val loss")
    ax2 = ax[1].twinx()
    ax2.plot(e, H["val_acc"], "g^-", label="val acc")
    ax2.set_ylabel("validation accuracy")
    ax[1].set(xlabel="epoch", ylabel="loss", title=f"final val acc = {H['val_acc'][-1]:.3f}")
    ax[1].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "mlp_sgd_training.png"), dpi=150)
    plt.close(fig)

    # ---------------- (d) gradient norms in a 10-layer MLP ----------------
    print("== Task 1(d): per-layer gradient norms ==")
    sizes = [3072] + [256] * 9 + [10]          # 10 Linear layers
    x, y = Xtr[:512], ytr[:512]
    configs = {"sigmoid (Xavier init)": ("sigmoid", "xavier"),
               "ReLU (Kaiming init)": ("relu", "kaiming"),
               "sigmoid (Kaiming init)": ("sigmoid", "kaiming"),
               "ReLU (Xavier init)": ("relu", "xavier")}
    norms = {}
    for name, (act, init) in configs.items():
        runs = []
        for s in range(5):                      # average over 5 random inits
            torch.manual_seed(s)
            net = MLP(sizes, act=act, init=init)
            net.loss(x, y)
            net.backward()
            runs.append([l.dW.norm().item() for l in net.linears()])
        norms[name] = np.mean(runs, 0).tolist()
        print(name, ["%.2e" % v for v in norms[name]])
    save_json(norms, os.path.join(od, "grad_norms.json"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = ["o-", "s-", "o--", "s--"]
    for (name, v), st in zip(norms.items(), styles):
        ax.semilogy(range(1, 11), v, st, label=name)
    ax.set(xlabel="layer (1 = closest to input)", ylabel=r"$\|\partial L/\partial W_\ell\|_F$",
           title="Per-layer gradient norm at initialisation, 10-layer MLP")
    ax.set_xticks(range(1, 11))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(od, "grad_norms.png"), dpi=150)
    plt.close(fig)
    print("Task 1 done ->", od)


if __name__ == "__main__":
    main()
