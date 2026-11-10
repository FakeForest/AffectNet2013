"""Evaluate a fixed equal-weight probability ensemble for workshop exercise 3."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from augment import to_device
from config import get_config
from data import load_data
from models import load_expression_model
import plots
from utils import get_device, predict, scores


MEMBERS = ("01_baseline", "02_native48", "04d_facenet")


def main():
    output = Path("results")
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as f:
        runs = {row["experiment"]: row for row in csv.DictReader(f)}
    paths = [runs[name]["checkpoint"] for name in MEMBERS]
    device = get_device()
    data, class_names = load_data(get_config([]))
    splits = to_device({name: data[name] for name in ("val", "test")}, device)
    models = [load_expression_model(path, device) for path in paths]
    if any(classes != class_names for _, _, classes in models):
        raise ValueError("Ensemble members must share the same ordered classes.")

    result = {"experiment": "03b_ensemble", "members": list(MEMBERS), "checkpoints": paths}
    for split, (x, y) in splits.items():
        predictions = [predict(model, x, y, cfg["img_size"], cfg.get("tta", True), batch_size=128)
                       for model, cfg, _ in models]
        probs = np.mean([member[0] for member in predictions], axis=0)
        labels = predictions[0][1]
        key = "validation" if split == "val" else split
        result.update({f"{key}_{metric}": value for metric, value in scores(probs, labels).items()})
        if split == "test":
            result["test_images"] = len(labels)
            matrix = confusion_matrix(labels, probs.argmax(1), labels=range(len(class_names)))
            result["confusion_matrix"] = matrix.tolist()
            result["class_names"] = class_names
            plots.confusion(matrix, class_names, output / "ensemble_confusion_matrix.png")

    (output / "ensemble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    winner = max(runs.values(), key=lambda run: float(run["validation_macro_f1"]))
    if result["validation_macro_f1"] > float(winner["validation_macro_f1"]):
        winner = {key: value for key, value in result.items() if key not in ("class_names", "confusion_matrix")}
    (output / "selected_model.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    print(f"Ensemble test macro-F1 {result['test_macro_f1']:.4f} | accuracy {result['test_accuracy']:.4f}")
    print(f"Selected by validation macro-F1: {winner['experiment']}")


if __name__ == "__main__":
    main()
