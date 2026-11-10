import argparse
from pathlib import Path

DEFAULTS = dict(
    # ---- data ----
    data_dir="data",            # where FER2013 is looked for / downloaded to
    labels="ferplus",           # "ferplus" (8 classes, cleaner labels) or "fer2013" (original 7 classes)
    ferplus_mode="majority",    # "majority" (one label per image) or "soft" (train on the annotators' vote shares)
    # ---- model ----
    model="resnet18",           # "resnet18", "alexnet" (as in the AffectNet paper) or "small_cnn" (from scratch)
    pretrained=True,            # start from ImageNet weights (ignored for small_cnn)
    img_size=128,               # FER2013 images are 48x48; pretrained networks work better on upscaled images
    # ---- training ----
    epochs=25,
    batch_size=128,
    lr=5e-4,
    weight_decay=1e-4,
    class_weighting="sqrt_inverse",  # "inverse" (paper-style), "sqrt_inverse" (gentler) or "none"
    label_smoothing=0.1,
    ema_decay=0.0,               # 0 disables exponential moving average of model weights
    augment=True,               # random flip / rotation / shift / brightness / erasing
    tta=True,                   # test-time augmentation: also predict on the mirrored face and average
    select_metric="macro_f1",   # validation score used to pick the best checkpoint: "macro_f1" or "accuracy"
    seed=42,
    # ---- output ----
    out_dir="runs",             # checkpoints, plots and results go to out_dir/<run name>/
)

HELP = dict(
    data_dir="folder containing FER2013 (fer2013.csv, the Kaggle zip, or train/ and test/ folders)",
    labels="ferplus: 8 classes, cleaner labels | fer2013: the original 7 classes",
    ferplus_mode="majority: one label per image | soft: train on the annotators' vote shares",
    model="alexnet is the AffectNet paper's network; resnet18_native is the 48px variant; facenet uses VGGFace2",
    pretrained="start from ImageNet weights (VGGFace2 for facenet)",
    img_size="size images are upscaled to (FER2013 images are 48x48)",
    epochs="number of passes over the training set",
    batch_size="images per training step",
    lr="peak learning rate",
    weight_decay="AdamW weight decay",
    class_weighting="weighted loss for class imbalance; inverse is the paper's choice",
    label_smoothing="label smoothing for the loss",
    ema_decay="decay per optimizer step for EMA weights used in validation and test (0 disables it)",
    augment="random flip / rotation / shift / brightness / erasing during training",
    tta="also predict on the mirrored face and average",
    select_metric="validation score used to pick the best checkpoint",
    seed="random seed",
    out_dir="results go to out_dir/<run name>/",
)

CHOICES = dict(
    labels=["ferplus", "fer2013"],
    ferplus_mode=["majority", "soft"],
    model=["resnet18", "resnet18_native", "alexnet", "small_cnn", "facenet"],
    class_weighting=["inverse", "sqrt_inverse", "none"],
    select_metric=["macro_f1", "accuracy"],
)


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="AffectNet-style facial expression baseline on FER2013")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        help_text = f"{HELP.get(name, '')} (default: {default})"
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=HELP.get(name))
        else:
            parser.add_argument(flag, type=type(default), default=default, choices=CHOICES.get(name), help=help_text)
    cfg = parser.parse_args(argv)
    if not 0 <= cfg.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if cfg.model == "resnet18_native" and cfg.img_size != 48:
        parser.error("resnet18_native requires --img-size 48")
    cfg.data_dir = Path(cfg.data_dir)
    cfg.run_dir = Path(cfg.out_dir) / run_name(cfg)
    return cfg


def run_name(cfg):
    mode = f"_{cfg.ferplus_mode}" if cfg.labels == "ferplus" else ""
    return f"{cfg.model}_{cfg.labels}{mode}_{cfg.img_size}px_{cfg.class_weighting}"
