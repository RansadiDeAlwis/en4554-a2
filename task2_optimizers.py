"""Task 2: optimisers.
(a) Custom SGD-momentum and Adam match torch.optim.SGD / torch.optim.Adam.
(b) Trajectories on f(x,y) = 0.5 (x^2 + 50 y^2).
(c) Small CNN, 3 optimisers x 3 learning rates; val acc vs lr, loss at best lr.
"""
import copy
from common import *
from models import SmallCNN
from optimizers import MySGD, MyAdam


# ---------------------------------------------------------------- (a)
def match_test(data, my_cls, torch_cls, kw, steps=100, label=""):
    """Train two identical copies, one per optimiser, on the same batches
    (float64, CPU so the arithmetic is deterministic)."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(8 * 16 * 16, 10)).double()
    m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
    o1, o2 = my_cls(m1.parameters(), **kw), torch_cls(m2.parameters(), **kw)
    X = data["xtr"][:2048].cpu().double()
    Y = data["ytr"][:2048].cpu()
    g = torch.Generator().manual_seed(1)
    l1s, l2s = [], []
    for _ in range(steps):
        idx = torch.randint(0, len(X), (64,), generator=g)
        for m, o, ls in ((m1, o1, l1s), (m2, o2, l2s)):
            o.zero_grad()
            loss = F.cross_entropy(m(X[idx]), Y[idx])
            loss.backward()
            o.step()
            ls.append(loss.item())
    diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))
    ok = all(torch.allclose(p1, p2, rtol=1e-7, atol=1e-10)
             for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"{label:35s} max|param diff| = {diff:.2e}  allclose = {ok}")
    return dict(config=label, max_abs_param_diff=diff, allclose=ok), l1s, l2s


# ---------------------------------------------------------------- (b)
def quad_traj(opt_cls, kw, start=(-10.0, 1.0), steps=100):
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = opt_cls([p], **kw)
    traj = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        f = 0.5 * (p[0] ** 2 + 50 * p[1] ** 2)
        f.backward()
        opt.step()
        traj.append(p.detach().clone().numpy())
    return np.array(traj)


def fquad(x, y):
    return 0.5 * (x ** 2 + 50 * y ** 2)


def main():
    p = base_parser("Task 2")
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    set_seed(args.seed)
    od = out_dir(args, "task2")
    data = load_cifar10(args)

    # ======================= (a) =======================
    print("== Task 2(a): custom optimisers vs torch.optim ==")
    tests = [
        (MySGD, torch.optim.SGD, dict(lr=0.05), "SGD (plain)"),
        (MySGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9), "SGD momentum 0.9"),
        (MySGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9, weight_decay=5e-4), "SGD momentum 0.9 + wd"),
        (MySGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9, nesterov=True), "SGD Nesterov 0.9"),
        (MyAdam, torch.optim.Adam, dict(lr=1e-3), "Adam lr=1e-3"),
        (MyAdam, torch.optim.Adam, dict(lr=1e-3, betas=(0.8, 0.99), weight_decay=1e-4), "Adam betas=(0.8,0.99) + wd"),
    ]
    rows, curves = [], {}
    for my, tc, kw, label in tests:
        r, l1, l2 = match_test(data, my, tc, kw, steps=20 if args.quick else 200, label=label)
        rows.append(r)
        curves[label] = (l1, l2)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(od, "optimizer_match.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, label in zip(ax, ["SGD momentum 0.9", "Adam lr=1e-3"]):
        l1, l2 = curves[label]
        a.plot(l1, lw=3, alpha=0.5, label="my implementation")
        a.plot(l2, "k--", lw=1, label="torch.optim")
        a.set(title=label, xlabel="step", ylabel="mini-batch loss")
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(od, "optimizer_match.png"), dpi=150)
    plt.close(fig)

    # ======================= (b) =======================
    print("== Task 2(b): trajectories on f = 0.5(x^2 + 50 y^2) ==")
    runs = {
        "SGD (lr=0.035)": (MySGD, dict(lr=0.035)),
        "SGD+momentum (lr=0.01, mu=0.9)": (MySGD, dict(lr=0.01, momentum=0.9)),
        "Adam (lr=0.3)": (MyAdam, dict(lr=0.3)),
    }
    trajs = {k: quad_traj(c, kw) for k, (c, kw) in runs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    xs, ys = np.linspace(-11, 6, 400), np.linspace(-2.2, 2.2, 400)
    XX, YY = np.meshgrid(xs, ys)
    ax[0].contour(XX, YY, fquad(XX, YY), levels=np.logspace(-2, 2.3, 18), cmap="viridis", linewidths=0.8)
    for (k, t), c in zip(trajs.items(), ["tab:red", "tab:blue", "tab:orange"]):
        ax[0].plot(t[:, 0], t[:, 1], "o-", ms=2.5, lw=1, color=c, label=k)
        ax[1].semilogy(fquad(t[:, 0], t[:, 1]) + 1e-16, color=c, label=k)
    ax[0].plot(0, 0, "k*", ms=12)
    ax[0].set(xlabel="x", ylabel="y", title=r"Trajectories on $f=\frac{1}{2}(x^2+50y^2)$, start (-10, 1), 100 steps")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].set(xlabel="iteration", ylabel="f(x, y)", title="objective vs iteration")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "quadratic_trajectories.png"), dpi=150)
    plt.close(fig)
    save_json({k: dict(final_xy=t[-1].tolist(), final_f=float(fquad(*t[-1])))
               for k, t in trajs.items()}, os.path.join(od, "quadratic_final.json"))
    for k, t in trajs.items():
        print(f"{k:35s} final (x,y)=({t[-1, 0]:.3e},{t[-1, 1]:.3e}) f={fquad(*t[-1]):.3e}")

    # ======================= (c) =======================
    print("== Task 2(c): small CNN, 3 optimisers x 3 learning rates ==")
    grid = {
        "SGD": (lambda ps, lr: MySGD(ps, lr=lr), [0.01, 0.05, 0.2]),
        "SGD+momentum": (lambda ps, lr: MySGD(ps, lr=lr, momentum=0.9), [0.001, 0.01, 0.05]),
        "Adam": (lambda ps, lr: MyAdam(ps, lr=lr), [1e-4, 1e-3, 1e-2]),
    }
    epochs = epochs_for(args, args.epochs)
    res_path = os.path.join(od, "lr_sweep.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name, (make, lrs) in grid.items():
        for lr in lrs:
            key = f"{name}@{lr:g}"
            if key in results and not args.quick:
                print("skip (already done):", key)
                continue
            set_seed(args.seed)
            model = SmallCNN().to(DEVICE)
            # constant lr and fp32 so only the optimiser differs
            H = train(model, data, epochs, make(model.parameters(), lr), bs=128,
                      amp=False, tag=key)
            results[key] = dict(opt=name, lr=lr, final_val_acc=H["val_acc"][-1],
                                best_val_acc=max(H["val_acc"]), hist=H)
            save_json(results, res_path)       # checkpoint after every run

    table = [dict(optimizer=v["opt"], lr=v["lr"], final_val_acc=v["final_val_acc"],
                  best_val_acc=v["best_val_acc"], diverged=v["hist"]["diverged"])
             for v in results.values()]
    pd.DataFrame(table).to_csv(os.path.join(od, "lr_sweep_table.csv"), index=False)
    print(pd.DataFrame(table).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = dict(zip(grid, ["tab:red", "tab:blue", "tab:orange"]))
    for name in grid:
        rs = sorted([v for v in results.values() if v["opt"] == name], key=lambda v: v["lr"])
        ax[0].semilogx([r["lr"] for r in rs], [r["final_val_acc"] for r in rs], "o-",
                       color=colors[name], label=name)
        best = max(rs, key=lambda r: r["final_val_acc"])
        il = best["hist"]["iter_loss"]
        ax[1].plot(np.arange(len(smooth(il))), smooth(il), color=colors[name],
                   label=f"{name} (lr={best['lr']:g})")
    ax[0].set(xlabel="learning rate (log scale)", ylabel=f"validation accuracy after {epochs} epochs",
              title="Learning-rate sensitivity")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()
    ax[1].set(xlabel="iteration", ylabel="training loss (moving avg 50)",
              title="Training loss at the best learning rate")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(od, "lr_sweep.png"), dpi=150)
    plt.close(fig)
    print("Task 2 done ->", od)


if __name__ == "__main__":
    main()
