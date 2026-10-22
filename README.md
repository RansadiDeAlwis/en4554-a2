# EN4554 Assignment 2: CNNs on CIFAR-10

## Files
| File | What it does |
|---|---|
| `common.py` | data split (45k/5k), normalisation, seeds, GPU augmentations (flip, crop, cutout, mixup), training loop |
| `models.py` | SmallCNN (T2), SmallBNCNN (T3a), plain/residual CIFAR nets (T3b), Task4Net (T4, 483k params) |
| `optimizers.py` | `MySGD` (momentum, Nesterov, weight decay) and `MyAdam` (bias-corrected) |
| `task1_backprop.py` | manual backprop MLP, gradient check, SGD training, gradient norms |
| `task2_optimizers.py` | optimizer match test, quadratic trajectories, lr sweep |
| `task3a_regularization.py` | 7 regularisation settings on 5k images |
| `task3b_residual.py` | plain vs residual, depths 8/20/56 |
| `task4_best_cnn.py` | optional: ablations + single test evaluation |
| `run_all.sh` / `job.sbatch` | run everything / Slurm job |
| `build_report.py` | builds `report.ipynb` from the results |

## Steps
```bash
# 1. environment
python -m venv ~/venvs/a2 && source ~/venvs/a2/bin/activate
pip install -r requirements.txt

# 2. smoke test (a few minutes; downloads CIFAR-10 into ./data)
EXTRA="--quick" OUT=results_quick ./run_all.sh

# 3. full run, survives logout
tmux new -s a2            # or: nohup ./run_all.sh > logs/all.out 2>&1 &
./run_all.sh              # Ctrl-b d to detach, `tmux attach -t a2` to come back
tail -f logs/task2.log    # watch progress

# 4. report
python build_report.py --out results --name "Your Name" --index "XXXXXXX" --github "https://github.com/you/repo"
jupyter nbconvert --to notebook --execute report.ipynb --output report.ipynb
# copy report.ipynb + results/ to your laptop, write the interpretations, then
# File > Save and Export as > PDF (or upload to Colab and print to PDF)
```

Run a single task: `python task3b_residual.py --out results --epochs 30 --depths 8 20 56`
(every script takes `--help`). Finished runs are saved after each configuration, so a
restarted script skips what is already done; delete `results/taskX/results.json` to redo.

Approximate T4 times: T1 ~5 min, T2 ~15 min, T3a ~10 min, T3b ~45 min, T4 ~30 min.
