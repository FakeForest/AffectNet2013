# AffectNet2013 Workshop

**Authors:** Yuelin Li and Xiaoyu Chen

An AffectNet-style facial expression classification baseline on FER2013 images with eight-class FER+ annotations. The workshop compares six fine-tuning and training configurations and an equal-weight ensemble on the same filtered private test split. Validation macro-F1 selects the checkpoints.

**Final report:** [WORKSHOP.pdf](WORKSHOP.pdf) is at the repository root. Edit [WORKSHOP.tex](WORKSHOP.tex) to update the report, or read the [Markdown version](WORKSHOP.md) directly on GitHub.

| Model | Test macro-F1 | Test accuracy |
| --- | ---: | ---: |
| ImageNet ResNet-18 baseline | 72.32% | 85.49% |
| VGGFace2 FaceNet | **73.99%** | 86.49% |
| Three-model ensemble | 73.29% | **87.23%** |

## Find the results

| What you want to see | Where to look |
| --- | --- |
| Compare the six individual models | [`results/comparison.csv`](results/comparison.csv) |
| Check ensemble scores and the final model choice | [Ensemble scores](results/ensemble.json), [ensemble confusion matrix](results/ensemble_confusion_matrix.png), and [model selected by validation macro-F1](results/selected_model.json) |
| Compare ResNet-18 and AlexNet during training | [Four learning curves in one plot](results/resnet18_vs_alexnet.png) |
| Inspect a specific run | [`results/experiments/`](results/experiments) |

| Exercise | Run folder |
| --- | --- |
| 1 · ResNet-18 baseline | [`01_baseline/`](results/experiments/01_baseline) |
| 2 · Native 48×48 ResNet-18 | [`02_native48/`](results/experiments/02_native48) |
| 3 · Longer schedule with EMA | [`03_tuned/`](results/experiments/03_tuned) |
| 4a · ResNet-18 from scratch | [`04a_scratch/`](results/experiments/04a_scratch) |
| 4b · ImageNet-pretrained AlexNet | [`04b_alexnet/`](results/experiments/04b_alexnet) |
| 4d · VGGFace2-pretrained FaceNet | [`04d_facenet/`](results/experiments/04d_facenet) |

Each run folder contains `config.json` for settings, `history.csv` for per-epoch metrics, and `test_results.json` for final test scores. The PNGs show class counts, learning curves, and the confusion matrix.

## Find the code

| Task | Files at the repository root |
| --- | --- |
| Run all six experiments or train one model | [`run_workshop.py`](run_workshop.py), [`train.py`](train.py), [`config.py`](config.py) |
| Load FER2013 and FER+ labels or change preprocessing | [`data.py`](data.py), [`augment.py`](augment.py) |
| Inspect the model architectures | [`models.py`](models.py) |
| Evaluate a checkpoint or the ensemble | [`evaluate.py`](evaluate.py), [`ensemble.py`](ensemble.py) |
| Predict the expression in your own photos | [`predict.py`](predict.py) |
| Check the metrics, plotting code, or dependencies | [`utils.py`](utils.py), [`plots.py`](plots.py), [`requirements.txt`](requirements.txt) |

## Compile the report

Run `tectonic WORKSHOP.tex` from the repository root. The LaTeX source uses the plot under `results/` and compiles to `WORKSHOP.pdf`.

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
