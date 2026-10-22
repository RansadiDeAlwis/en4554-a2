"""Task 3(a): regularisation on a 5k-image training subset.
baseline / weight decay / dropout / flip+crop / cutout / mixup / all combined.
Train accuracy is measured on the CLEAN (un-augmented, un-mixed) 5k images in
eval mode, so the train-val gap is comparable across methods."""
from common import *
from models import SmallBNCNN
from optimizers import MySGD

# >>> REPORT: reg_configs
CONFIGS = {
    "baseline":     dict(),
    "weight_decay": dict(wd=5e-4),
    "dropout":      dict(dropout=0.5),
    "flip+crop":    dict(aug=["flip", "crop"]),
    "cutout":       dict(aug=["cutout"]),
    "mixup":        dict(mixup=0.4),
    "all":          dict(wd=5e-4, dropout=0.5, aug=["flip", "crop", "cutout"], mixup=0.4),
}
# <<< REPORT


def main():
    p = base_parser("Task 3a")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-train", type=int, default=5000)
    args = p.parse_args()
    set_seed(args.seed)
    od = out_dir(args, "task3a")
    data = load_cifar10(args)
    n = min(args.n_train, len(data["xtr"]))
    xs, ys = data["xtr"][:n], data["ytr"][:n]       # 5k images (split is random)
    epochs = epochs_for(args, args.epochs)
    amp = not args.no_amp

    res_path = os.path.join(od, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name, cfg in CONFIGS.items():
        if name in results and not args.quick:
            print("skip (already done):", name)
            continue
        set_seed(args.seed)
        model = SmallBNCNN(dropout=cfg.get("dropout", 0.0)).to(DEVICE)
        opt = MySGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=cfg.get("wd", 0.0))
        steps = epochs * math.ceil(n / 128)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        H = train(model, data, epochs, opt, sched, bs=128, aug=make_aug(cfg.get("aug")),
                  mixup_alpha=cfg.get("mixup", 0.0), amp=amp, xtr=xs, ytr=ys,
                  clean_train_eval=True, tag=name)
        results[name] = dict(cfg=cfg, hist=H)
        save_json(results, res_path)

    import pandas as pd
    rows = []
    for name, r in results.items():
        H = r["hist"]
        tr, va = H["train_acc_clean"][-1], H["val_acc"][-1]
        rows.append(dict(method=name, train_acc=round(tr, 4), val_acc=round(va, 4),
                         gap=round(tr - va, 4), best_val_acc=round(max(H["val_acc"]), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "table.csv"), index=False)
    print(df.to_string(index=False))

    names = list(results)
    cols = 4
    rws = math.ceil(len(names) / cols)
    fig, axs = plt.subplots(rws, cols, figsize=(4 * cols, 3.2 * rws), sharey=True)
    for a, name in zip(axs.flat, names):
        H = results[name]["hist"]
        e = np.arange(1, len(H["val_acc"]) + 1)
        a.plot(e, H["train_acc_clean"], label="train (clean)")
        a.plot(e, H["val_acc"], label="validation")
        a.set_title(f"{name}: gap {H['train_acc_clean'][-1] - H['val_acc'][-1]:.3f}")
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    for a in list(axs.flat)[len(names):]:
        a.axis("off")
    axs.flat[0].set_ylabel("accuracy")
    axs.flat[0].legend()
    fig.suptitle("Task 3(a): train vs validation accuracy, 5k training images")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "curves_grid.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in names:
        H = results[name]["hist"]
        e = np.arange(1, len(H["val_acc"]) + 1)
        ax[0].plot(e, H["val_acc"], label=name)
        ax[1].plot(e, H["val_loss"], label=name)
    ax[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
    ax[1].set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "val_curves.png"), dpi=150)
    plt.close(fig)
    print("Task 3a done ->", od)


if __name__ == "__main__":
    main()
