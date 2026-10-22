"""Task 4 (optional): best CNN with <= 500k parameters and <= 20 epochs.
Every run is tuned/compared on the VALIDATION split. Only the selected model is
evaluated on the test set, exactly once (a guard file prevents re-evaluation)."""
from common import *
from models import Task4Net
from optimizers import MySGD

MAX_PARAMS, MAX_EPOCHS = 500_000, 20

# >>> REPORT: task4_configs
FULL = dict(shortcut=True, aug=["flip", "crop", "cutout"], cutout_size=8,
            opt="sgd", lr=0.1, wd=5e-4, label_smoothing=0.1, bs=256)
ABLATIONS = {                                  # each changes ONE design choice
    "full":              {},
    "no_shortcut":       dict(shortcut=False),
    "no_augmentation":   dict(aug=[]),
    "flip_crop_only":    dict(aug=["flip", "crop"]),
    "no_weight_decay":   dict(wd=0.0),
    "adamw_instead_sgd": dict(opt="adamw", lr=2e-3, wd=5e-2),
    "no_label_smoothing": dict(label_smoothing=0.0),
}
# <<< REPORT


def one_cycle(opt, total_steps, warm=0.25):
    """Linear warm-up to the peak lr for 25% of the steps, then linear decay to 0."""
    w = int(warm * total_steps)
    f = lambda s: (s + 1) / w if s < w else max(0.0, (total_steps - s) / (total_steps - w))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def build(cfg):
    model = Task4Net(shortcut=cfg["shortcut"]).to(DEVICE).to(memory_format=torch.channels_last)
    n = count_params(model)
    assert n <= MAX_PARAMS, f"{n} parameters > {MAX_PARAMS}"
    if cfg["opt"] == "sgd":
        opt = MySGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True,
                    weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    return model, opt, n


def main():
    p = base_parser("Task 4")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--runs", nargs="+", default=list(ABLATIONS))
    p.add_argument("--evaluate-test", action="store_true",
                   help="after all ablations, evaluate the best one on the test set ONCE")
    args = p.parse_args()
    assert args.epochs <= MAX_EPOCHS
    set_seed(args.seed)
    od = out_dir(args, "task4")
    data = load_cifar10(args, load_test=args.evaluate_test)
    epochs = epochs_for(args, args.epochs)
    amp = not args.no_amp

    res_path = os.path.join(od, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name in args.runs:
        if name in results and not args.quick:
            print("skip (already done):", name)
            continue
        cfg = {**FULL, **ABLATIONS[name]}
        set_seed(args.seed)
        model, opt, n = build(cfg)
        print(f"{name}: {n:,} params, cfg={cfg}")
        steps = epochs * math.ceil(len(data["xtr"]) / cfg["bs"])
        t0 = time.time()
        H = train(model, data, epochs, opt, one_cycle(opt, steps), bs=cfg["bs"],
                  aug=make_aug(cfg["aug"], cfg["cutout_size"]),
                  label_smoothing=cfg["label_smoothing"], amp=amp, tag=name)
        torch.save(model.state_dict(), os.path.join(od, f"{name}.pt"))
        results[name] = dict(cfg=cfg, params=n, train_time_s=time.time() - t0,
                             final_val_acc=H["val_acc"][-1], hist=H)
        save_json(results, res_path)

    import pandas as pd
    rows = [dict(run=k, params=v["params"], final_val_acc=round(v["final_val_acc"], 4),
                 delta_vs_full=round(v["final_val_acc"] - results.get("full", v)["final_val_acc"], 4),
                 minutes=round(v["train_time_s"] / 60, 1)) for k, v in results.items()]
    df = pd.DataFrame(rows).sort_values("final_val_acc", ascending=False)
    df.to_csv(os.path.join(od, "ablation_table.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, v in results.items():
        e = np.arange(1, len(v["hist"]["val_acc"]) + 1)
        ax[0].plot(e, v["hist"]["val_acc"], label=k, lw=2 if k == "full" else 1)
    ax[0].set(xlabel="epoch", ylabel="validation accuracy", title="Ablation: validation accuracy")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    d = df.sort_values("final_val_acc")
    ax[1].barh(d["run"], d["final_val_acc"])
    lo = d["final_val_acc"].min()
    ax[1].set_xlim(max(0, lo - 0.03), min(1, d["final_val_acc"].max() + 0.01))
    ax[1].set(xlabel="final validation accuracy", title="Ablation summary")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "ablation.png"), dpi=150)
    plt.close(fig)

    # ---------------- test set: evaluated ONCE ----------------
    if args.evaluate_test:
        guard = os.path.join(od, "TEST_EVALUATED.json")
        if os.path.exists(guard) and not args.quick:
            print("Test set already evaluated once; result:", open(guard).read())
            return
        best = max(results, key=lambda k: results[k]["final_val_acc"])   # chosen on VAL
        cfg = results[best]["cfg"]
        model, _, n = build(cfg)
        model.load_state_dict(torch.load(os.path.join(od, f"{best}.pt"), map_location=DEVICE))
        test_loss, test_acc = evaluate(model, data["xte"], data["yte"], amp=amp)
        out = dict(selected_run=best, params=n, val_acc=results[best]["final_val_acc"],
                   test_acc=test_acc, test_loss=test_loss, cfg=cfg)
        save_json(out, guard)
        print("FINAL TEST RESULT:", out)
    print("Task 4 done ->", od)


if __name__ == "__main__":
    main()
