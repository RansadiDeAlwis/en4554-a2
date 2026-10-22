"""Builds report.ipynb from the results folder: title, key code (pulled from the
'# >>> REPORT' blocks in the scripts), figures, tables, and an interpretation
cell after every result for YOU to fill in. Then export the notebook to PDF."""
import os, re, json, argparse
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def snippet(fname, name):
    src = open(os.path.join(HERE, fname)).read()
    m = re.search(rf"# >>> REPORT: {name}\n(.*?)# <<< REPORT", src, re.S)
    return m.group(1).rstrip() if m else f"# snippet {name} not found in {fname}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="YOUR NAME")
    ap.add_argument("--index", default="YOUR INDEX NUMBER")
    ap.add_argument("--github", default="https://github.com/YOUR_USERNAME/en4554-a2")
    a = ap.parse_args()
    R = a.out
    nb = nbf.v4.new_notebook()
    C = nb.cells
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code_listing = lambda f, n: C.append(nbf.v4.new_markdown_cell(f"```python\n{snippet(f, n)}\n```"))
    run = lambda s: C.append(nbf.v4.new_code_cell(s))
    img = lambda p: run(f"display(Image(filename=os.path.join(R, {p!r})))")
    tab = lambda p: run(f"display(pd.read_csv(os.path.join(R, {p!r})))")
    interp = lambda hint: md(f"**Interpretation.** _TODO: write this yourself. Points to address: {hint}_")

    md(f"# EN4554 Assignment 2: CNNs\n**Name:** {a.name}  \n**Index:** {a.index}  \n"
       f"**GitHub:** {a.github}\n\n**Acknowledgement of AI tools:** _TODO: state how you "
       f"used Claude/Copilot (e.g. code scaffolding), and that you understand all the code._\n\n"
       f"Experiments were run as scripts on a GPU server; this notebook shows the key code "
       f"and the saved results.")
    run(f"import os, json\nimport pandas as pd\nfrom IPython.display import Image, display\n"
        f"pd.set_option('display.precision', 4)\nR = {os.path.abspath(R)!r}")
    md("## Setup\n45k/5k split of the CIFAR-10 training set with a fixed seed, per-channel "
       "normalisation with the 45k training statistics, seeds fixed. Test set only used in Task 4.")
    code_listing("common.py", "data")

    md("## Task 1: Backpropagation\n### (a) Layers with explicit forward/backward")
    code_listing("task1_backprop.py", "layers")
    md("### (b) Gradient check")
    code_listing("task1_backprop.py", "gradcheck")
    tab("task1/gradcheck.csv")
    interp("size of the relative errors vs float64 precision; why FD error (~1e-9) is larger "
           "than autograd error (~1e-16): truncation O(eps^2) + round-off O(machine_eps/eps).")
    md("### (c) SGD training on flattened CIFAR-10")
    img("task1/mlp_sgd_training.png")
    run("print(json.load(open(os.path.join(R,'task1/mlp_sgd_summary.json'))))")
    interp("final validation accuracy; does the loss still decrease; train/val loss gap; "
           "why an MLP on raw pixels is limited (no translation invariance, no weight sharing).")
    md("### (d) Per-layer gradient norms in a 10-layer MLP")
    img("task1/grad_norms.png")
    interp("sigmoid derivative <= 1/4 so the backward signal shrinks geometrically toward the "
           "input (vanishing gradients); ReLU derivative is 0 or 1 and Kaiming init keeps the "
           "variance constant; what the mismatched-init curves show.")

    md("## Task 2: Optimizers\n### (a) Custom SGD-momentum and Adam")
    code_listing("optimizers.py", "optimizers")
    tab("task2/optimizer_match.csv")
    img("task2/optimizer_match.png")
    interp("parameters agree to ~1e-17 after 200 steps, i.e. identical up to rounding.")
    md("### (b) Trajectories on f(x,y) = (x^2 + 50 y^2)/2")
    img("task2/quadratic_trajectories.png")
    interp("condition number 50; SGD stable only for lr < 2/50, zig-zags in y while crawling in x; "
           "momentum accelerates along x and damps the oscillation; Adam rescales each coordinate "
           "by its gradient magnitude so it moves diagonally at first.")
    md("### (c) Small CNN: 3 optimizers x 3 learning rates")
    code_listing("models.py", "smallcnn")
    tab("task2/lr_sweep_table.csv")
    img("task2/lr_sweep.png")
    interp("which optimizer is most / least sensitive to lr; speed of the loss decrease at the "
           "best lr; generalisation differences.")

    md("## Task 3: Regularization and residual connections\n### (a) Regularization on 5k images")
    code_listing("common.py", "augment")
    code_listing("task3a_regularization.py", "reg_configs")
    tab("task3a/table.csv")
    img("task3a/curves_grid.png")
    img("task3a/val_curves.png")
    interp("baseline overfits (train ~100%, large gap); which methods shrink the gap and which "
           "actually raise val accuracy; why mixup/all lower train accuracy; are effects additive?")
    md("### (b) Plain vs residual networks (depth 8, 20, 56)")
    code_listing("models.py", "resnet")
    tab("task3b/table.csv")
    img("task3b/plain_vs_residual.png")
    interp("deeper PLAIN nets have HIGHER training loss (degradation problem; not overfitting, "
           "since it is the training loss); residual nets improve with depth because the identity "
           "path makes it easy to represent near-identity mappings and gives gradients a direct path.")

    if os.path.exists(os.path.join(R, "task4/ablation_table.csv")):
        md("## Task 4: Best CNN (<= 500k parameters, <= 20 epochs)")
        code_listing("models.py", "task4net")
        code_listing("task4_best_cnn.py", "task4_configs")
        tab("task4/ablation_table.csv")
        img("task4/ablation.png")
        if os.path.exists(os.path.join(R, "task4/TEST_EVALUATED.json")):
            run("print(json.load(open(os.path.join(R,'task4/TEST_EVALUATED.json'))))")
        interp("contribution of each design choice; selection was on validation only; test "
               "accuracy reported once.")

    nbf.write(nb, os.path.join(HERE, "report.ipynb"))
    print("wrote", os.path.join(HERE, "report.ipynb"))


if __name__ == "__main__":
    main()
