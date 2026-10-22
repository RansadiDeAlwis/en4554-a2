"""Task 3(b): plain vs residual CIFAR networks of depth 8, 20, 56.
Both families are identical except for the (parameter-free) shortcut."""
from common import *
from models import CifarResNet
from optimizers import MySGD


def main():
    p = base_parser("Task 3b")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--depths", type=int, nargs="+", default=[8, 20, 56])
    p.add_argument("--lr", type=float, default=0.1)
    args = p.parse_args()
    set_seed(args.seed)
    od = out_dir(args, "task3b")
    data = load_cifar10(args)
    epochs = epochs_for(args, args.epochs)
    amp = not args.no_amp
    aug = make_aug(["flip", "crop"])           # standard CIFAR augmentation

    res_path = os.path.join(od, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for depth in args.depths:
        for residual in [False, True]:
            key = f"{'resnet' if residual else 'plain'}-{depth}"
            if key in results and not args.quick:
                print("skip (already done):", key)
                continue
            set_seed(args.seed)
            model = CifarResNet(depth, residual=residual).to(DEVICE)
            n_params = count_params(model)
            print(f"{key}: {n_params:,} parameters")
            opt = MySGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            it = math.ceil(len(data["xtr"]) / 128)
            # step decay /10 at 50% and 75% of training (as in He et al.)
            sched = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[int(0.5 * epochs) * it, int(0.75 * epochs) * it], gamma=0.1)
            H = train(model, data, epochs, opt, sched, bs=128, aug=aug, amp=amp, tag=key)
            results[key] = dict(depth=depth, residual=residual, params=n_params, hist=H)
            save_json(results, res_path)

    import pandas as pd
    rows = [dict(model=k, params=v["params"], final_train_loss=round(v["hist"]["train_loss"][-1], 4),
                 final_train_acc=round(v["hist"]["train_acc"][-1], 4),
                 final_val_acc=round(v["hist"]["val_acc"][-1], 4)) for k, v in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "table.csv"), index=False)
    print(df.to_string(index=False))

    depths = sorted({v["depth"] for v in results.values()})
    colors = dict(zip(depths, ["tab:green", "tab:blue", "tab:red", "tab:purple", "tab:brown"]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for k, v in results.items():
        H, d = v["hist"], v["depth"]
        ls = "-" if v["residual"] else "--"
        e = np.arange(1, len(H["train_loss"]) + 1)
        ax[0].plot(e, H["train_loss"], ls, color=colors[d], label=k)
        ax[1].plot(np.arange(len(smooth(H["iter_loss"], 100))), smooth(H["iter_loss"], 100),
                   ls, color=colors[d], label=k, lw=1)
        ax[2].plot(e, [1 - a for a in H["val_acc"]], ls, color=colors[d], label=k)
    ax[0].set(xlabel="epoch", ylabel="training loss (epoch mean)", yscale="log",
              title="Training loss: plain (dashed) vs residual (solid)")
    ax[1].set(xlabel="iteration", ylabel="training loss (moving avg 100)", yscale="log",
              title="Training loss per iteration")
    ax[2].set(xlabel="epoch", ylabel="validation error", title="Validation error")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "plain_vs_residual.png"), dpi=150)
    plt.close(fig)
    print("Task 3b done ->", od)


if __name__ == "__main__":
    main()
