"""Run the six FER+ workshop experiments sequentially on a single GPU."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


RUNS = {
    "01_baseline": [],
    "02_native48": ["--model", "resnet18_native", "--img-size", "48"],
    "03_tuned": ["--epochs", "55", "--lr", "0.0003", "--ema-decay", "0.995",
                 "--ferplus-mode", "soft", "--label-smoothing", "0.02"],
    "04a_scratch": ["--no-pretrained"],
    "04b_alexnet": ["--model", "alexnet", "--epochs", "30"],
    "04d_facenet": ["--model", "facenet", "--img-size", "160", "--epochs", "30",
                    "--lr", "0.0002"],
}


def summarize():
    from config import get_config

    rows = []
    histories = {}
    for name, flags in RUNS.items():
        config = get_config(["--out-dir", f"runs/{name}", *flags])
        path = config.run_dir
        result_file = path / "test_results.json"
        if not result_file.exists():
            continue
        with result_file.open(encoding="utf-8") as f:
            result = json.load(f)
        with (path / "history.csv").open(encoding="utf-8", newline="") as f:
            history = list(csv.DictReader(f))
        histories[name] = history
        best = history[result["epoch"] - 1]
        rows.append({
            "experiment": name,
            "epochs": len(history),
            "best_epoch": result["epoch"],
            "validation_macro_f1": float(best["macro_f1"]),
            "validation_accuracy": float(best["accuracy"]),
            "test_macro_f1": result["macro_f1"],
            "test_accuracy": result["accuracy"],
            "test_images": result["test_images"],
            "checkpoint": result["checkpoint"],
        })
    if not rows:
        return

    baseline = next((row for row in rows if row["experiment"] == "01_baseline"), None)
    for row in rows:
        for metric in ("test_macro_f1", "test_accuracy"):
            row[f"{metric}_delta_pp"] = (
                round(100 * (row[metric] - baseline[metric]), 4) if baseline else ""
            )

    output = Path("results")
    output.mkdir(exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_run = max(rows, key=lambda row: row["validation_macro_f1"])
    (output / "best_model.json").write_text(json.dumps(best_run, indent=2), encoding="utf-8")

    if {"01_baseline", "04b_alexnet"} <= histories.keys():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
        metrics = ("train_loss", "val_loss", "accuracy", "macro_f1")
        titles = ("Training loss", "Validation loss", "Validation accuracy", "Validation macro-F1")
        for ax, key, title in zip(axes.flat, metrics, titles):
            for name, label in (("01_baseline", "ResNet-18"), ("04b_alexnet", "AlexNet")):
                history = histories[name]
                ax.plot([int(row["epoch"]) for row in history],
                        [float(row[key]) for row in history], label=label)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.savefig(output / "resnet18_vs_alexnet.png", dpi=180)
        plt.close(fig)


def main():
    os.chdir(Path(__file__).resolve().parent)
    import torch
    from config import get_config

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the workshop batch.")
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    failed = []
    for name, flags in RUNS.items():
        args = ["--out-dir", f"runs/{name}", *flags]
        result = get_config(args).run_dir / "test_results.json"
        if result.exists():
            print(f"{name} already finished: {result}", flush=True)
            summarize()
            continue
        command = [sys.executable, "-u", "train.py", *args]
        log_path = logs / f"{name}.log"
        print(f"Running {name}; see {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            print(subprocess.list2cmdline(command), file=log, flush=True)
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode or not result.exists():
            print(f"{name} failed with code {process.returncode}; see {log_path}", flush=True)
            failed.append(name)
            continue
        summarize()
        print(f"Finished {name}: {result}", flush=True)
    if failed:
        raise SystemExit(f"Failed experiments: {', '.join(failed)}")
    print("All six workshop runs completed.", flush=True)


if __name__ == "__main__":
    main()
