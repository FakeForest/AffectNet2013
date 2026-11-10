# AffectNet2013 Workshop

An AffectNet-style facial expression classification baseline on FER2013 images with eight-class FER+ annotations. The workshop compares six fine-tuning and training configurations and an equal-weight ensemble on the same filtered private test split. Validation macro-F1 selects the checkpoints.

| Model | Test macro-F1 | Test accuracy |
| --- | ---: | ---: |
| ImageNet ResNet-18 baseline | 72.32% | 85.49% |
| VGGFace2 FaceNet | **73.99%** | 86.49% |
| Three-model ensemble | 73.29% | **87.23%** |

[Workshop report](WORKSHOP.md) · [All single-model results](results/comparison.csv) · [Per-epoch histories](results/experiments) · [ResNet-18 and AlexNet curves](results/resnet18_vs_alexnet.png)

## Reproduce the workshop

Use Python 3.12 and a CUDA-enabled PyTorch installation. The following commands use the versions from the original Windows RTX 4090 run. Activate your virtual environment first.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt requests tqdm
python -m pip install --no-deps facenet-pytorch==2.6.0
```

The loader downloads the original FER2013 archive and Microsoft's FER+ labels into `data/`. You can also place the original `fer2013.csv` there before training. The order of its rows is used to align images with the FER+ labels. Pretrained weights are downloaded by torchvision and FaceNet on first use.

```sh
python run_workshop.py
python ensemble.py
```

The first command trains the six experiments in sequence and writes new checkpoints under `runs/`. The second evaluates the ensemble using those checkpoints. The tracked files under `results/experiments/` are snapshots from the original run.

After training, run the ensemble on a local photo by passing its three member checkpoints. You can pass a single checkpoint to use an individual model.

```sh
python predict.py --checkpoint runs/01_baseline/resnet18_ferplus_majority_128px_sqrt_inverse/best.pt --checkpoint runs/02_native48/resnet18_native_ferplus_majority_48px_sqrt_inverse/best.pt --checkpoint runs/04d_facenet/facenet_ferplus_majority_160px_sqrt_inverse/best.pt --output-dir results/photos your_photo.jpg
```

Dataset files, model weights, and workshop photos are kept outside the public repository.
