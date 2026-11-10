import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import plots
from augment import prepare
from models import load_expression_model
from utils import get_device


def find_face(gray):
    """Return (x, y, w, h) of the largest face, or the whole image if none is found."""
    faces = []
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not detector.empty():
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    except Exception:
        pass
    if len(faces) == 0:
        print("  no face found (or face detector unavailable), using the whole image")
        return 0, 0, gray.shape[1], gray.shape[0]
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict facial expressions in photos")
    parser.add_argument("--checkpoint", required=True, action="append",
                        help="path to best.pt; repeat to average model probabilities")
    parser.add_argument("--output-dir", type=Path, help="save plots and a predictions.json summary here")
    parser.add_argument("images", nargs="+", help="photo files")
    args = parser.parse_args(argv)

    device = get_device()
    models = [load_expression_model(path, device) for path in args.checkpoint]
    class_names = models[0][2]
    if any(classes != class_names for _, _, classes in models):
        raise ValueError("All checkpoints must use the same ordered classes.")

    predictions = []
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        print(path)
        img = cv2.imread(str(path))
        if img is None:
            print("  could not read this file")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = find_face(gray)
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
        xb = torch.from_numpy(face).view(1, 1, 48, 48).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                    enabled=device.type == "cuda"):
            member_probs = []
            for model, cfg, _ in models:
                xin = prepare(xb, cfg["img_size"], train=False)
                logits = model(xin).float()
                if cfg.get("tta", True):
                    logits = (logits + model(xin.flip(3)).float()) / 2
                member_probs.append(logits.softmax(1)[0])
            probs = torch.stack(member_probs).mean(0).cpu().numpy()
        for i in np.argsort(-probs)[:3]:
            print(f"  {class_names[i]:9s} {probs[i]:.1%}")
        output_dir = args.output_dir or Path(path).parent
        out = output_dir / (Path(path).stem + "_prediction.png")
        plots.photo_prediction(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (x, y, w, h), face, probs, class_names, out)
        print("  saved", out)
        predictions.append({
            "image": path,
            "face_box_xywh": [x, y, w, h],
            "predicted_class": class_names[int(probs.argmax())],
            "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
            "plot": str(out),
        })

    if args.output_dir is not None:
        summary = {"checkpoints": args.checkpoint, "predictions": predictions}
        (args.output_dir / "predictions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
