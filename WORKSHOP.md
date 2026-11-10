# AffectNet-style expression classification on FER2013

**Authors:** Yuelin Li and Xiaoyu Chen

## Protocol

The workshop adapts AffectNet's weighted expression classifier to the FER2013 images with Microsoft's FER+ annotations. The source CSV matches the original Kaggle checksum. The eight classes are neutral, happy, surprise, sad, angry, disgust, fear, and contempt. The original Training, PublicTest, and PrivateTest partitions provide 26,940 training, 3,396 validation, and 3,376 test examples after the supplied FER+ filters exclude removed images, ambiguous ties, and examples dominated by unknown or non-face votes. The task is expression classification. FER2013 does not supply valence or arousal labels. [AffectNet paper](https://arxiv.org/abs/1708.03985) · [Microsoft FER+ annotations](https://github.com/microsoft/FERPlus)

Each run uses seed 42, AdamW, a OneCycle schedule, class weights proportional to the inverse square root of class frequency, the supplied training augmentations, and horizontal-flip test-time augmentation. Validation macro-F1 selects the checkpoint for each run and the final model for the photos. Macro-F1 gives each class equal weight, including disgust and contempt, which are scarce in the training split. Accuracy measures the fraction of all examples classified correctly. Results are from one run per configuration.

## Experiment settings

| Exercise | Run | Input | Initialization | Epochs | Peak LR | Main change |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 1 | ResNet-18 baseline | 128×128 | ImageNet | 25 | 0.0005 | Supplied training defaults |
| 2 | Native-resolution ResNet-18 | 48×48 | ImageNet | 25 | 0.0005 | A smaller input stem |
| 3 | Tuned ResNet-18 | 128×128 | ImageNet | 55 | 0.0003 | EMA, vote distributions, less label smoothing |
| 4a | ResNet-18 from scratch | 128×128 | Random | 25 | 0.0005 | Same architecture and recipe as the baseline |
| 4b | AlexNet | 128×128 | ImageNet | 30 | 0.0005 | AlexNet backbone |
| 4d | FaceNet | 160×160 | VGGFace2 | 30 | 0.0002 | Inception-ResNet-v1 with an eight-class head |

In exercise 2, a 3×3 convolution with stride 1 uses the center of the pretrained 7×7 filters as initialization. A 2×2 pooling layer reduces the map from 48×48 to 24×24 before the first residual stage. A standard ResNet-18 at 48×48 would reduce it to 12×12 by that stage. The native variant keeps the original image samples and retains a 3×3 map before global average pooling.

In exercise 3, the 55-epoch schedule gives fine-tuning more optimizer updates at a lower peak learning rate. EMA uses a decay of 0.995 per batch to reduce sensitivity to short-term changes in the weights. FER+ vote shares supply a distribution of plausible labels for each image. Label smoothing drops from 0.10 to 0.02 because the vote shares already express uncertainty. Both training targets produce the same hard labels for validation and test scoring. These changes were tested as one recipe, so the run estimates their joint effect. [FER+ label-distribution paper](https://arxiv.org/abs/1608.01041)

In exercise 4d, FaceNet's Inception-ResNet-v1 starts with VGGFace2 face identity weights. A model trained on identity across poses and lighting may transfer facial structure and appearance features to expression recognition. The identity classifier is replaced with an eight-class head, and the full network is fine-tuned using FaceNet's fixed input standardization. This comparison also changes the backbone, input size, and learning rate. The baseline versus 4a is the controlled comparison for the effect of ImageNet initialization. [FaceNet PyTorch model and weights](https://github.com/timesler/facenet-pytorch) · [VGGFace2 paper](https://arxiv.org/abs/1710.08092)

## Published accuracy references

| Label protocol | Paper or model | Reported accuracy |
| --- | --- | ---: |
| FER2013, seven original classes | 12-layer CNN, *IET Computers & Digital Techniques*, 2025 | 99.20% maximum in the abstract and for the happy class in §4.1 |
| FER2013, seven original classes | ResNet50–Transformer, *Discover Computing*, 2026 | 89.70% as claimed in the paper |
| FER2013, seven original classes | Complementary global–local ensemble, *Information*, 2026 | 76.51% on a 7,178-image split |
| FER2013, seven original classes | SFER-MDFAE, *Sensors*, 2024 | 76.18% |
| FER2013, seven original classes | VGG-based single network without extra training images, 2021 | 73.28% |
| FER+ | PAtt-Lite comparison, ACIIDS, 2025 | 96.20% in the abstract |
| FER+, eight classes | FER-EMFormer, 2026 | 95.78% |

The highest FER2013 headline figure in the reviewed papers is 99.20%. Section 4.1 gives that number for the happy class and separately gives 98.20% as a mean across classes. Section 4.2 also applies a confidence filter for its detailed metrics. The 89.70% paper calls its result overall accuracy, but its Figure 3 discussion lists at least 991 distinct errors across seven cells, which exceeds the roughly 370 errors compatible with that score on the 3,589-image official private test partition. Both headline numbers require care when interpreting the benchmark. [IET paper, §§4.1–4.2](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cdt2/1845378) · [89.70% claim and Figure 3 discussion](https://link.springer.com/article/10.1007/s10791-026-10715-8)

SFER-MDFAE provides a seven-class reference at 76.18%. The 76.51% ensemble study uses the 7,178-image Kaggle folder test split, and the 73.28% paper documents a single-network result without extra training images. The seven-class results use different labels and sample eligibility from this workshop. [SFER-MDFAE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11511472/) · [76.51% ensemble paper](https://www.mdpi.com/2078-2489/17/10/982) · [73.28% single-network paper](https://arxiv.org/abs/2105.03588)

The PAtt-Lite comparison reports 96.20% FER+ accuracy and 68.40% balanced accuracy. Its full evaluation protocol was unavailable in the public preview. FER-EMFormer explicitly evaluates eight expressions and uses 224×224 inputs and MS-Celeb-1M pretraining. The local loader uses its own eligibility filters, so these figures provide context for the workshop accuracy. [PAtt-Lite comparison](https://link.springer.com/chapter/10.1007/978-981-96-5887-9_18) · [FER-EMFormer paper](https://link.springer.com/article/10.1007/s44443-026-00740-4)

## Results and interpretation

All rows use the same 3,376 private-test examples. Deltas are percentage points relative to the 128×128 ResNet-18 baseline.

| Exercise | Run | Best epoch | Val macro-F1 | Test macro-F1 | Test accuracy | Δ macro-F1 | Δ accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ResNet-18 baseline | 21 | 76.37% | 72.32% | 85.49% | 0.00 | 0.00 |
| 2 | Native 48×48 ResNet-18 | 13 | 75.05% | 70.89% | 83.23% | −1.43 | −2.25 |
| 3 | Longer schedule, soft targets, EMA | 16 | 75.32% | 69.50% | 84.57% | −2.82 | −0.92 |
| 3b | Equal-weight ensemble | — | **77.24%** | 73.29% | **87.23%** | +0.97 | **+1.75** |
| 4a | ResNet-18 from scratch | 23 | 72.76% | 67.78% | 82.49% | −4.54 | −2.99 |
| 4b | AlexNet | 26 | 69.28% | 68.24% | 82.32% | −4.08 | −3.17 |
| 4d | VGGFace2 FaceNet | 25 | 76.50% | **73.99%** | 86.49% | **+1.67** | +1.01 |

The baseline reached 72.32% test macro-F1 and 85.49% test accuracy. Happy achieved 94.9% F1, compared with 45.0% for disgust and 48.0% for contempt. The private test set has 22 disgust examples and 26 contempt examples, so a few predictions can noticeably change macro-F1.

The native 48×48 run reached 70.89% test macro-F1 and 83.23% accuracy, down 1.43 and 2.25 percentage points. Its F1 improved for disgust (45.0% to 47.8%) and contempt (48.0% to 51.3%) but dropped for fear (62.1% to 57.0%). The overall result reflects both the smaller input and the changed stem. Resizing to 128×128 gives pretrained filters a denser grid on which to operate, even though it interpolates the existing pixels.

The 55-epoch recipe reached 69.50% test macro-F1 and 84.57% accuracy, down 2.82 and 0.92 percentage points. Validation macro-F1 peaked at epoch 16 even though training loss continued to fall. Disgust recall rose from 40.9% to 59.1%, but precision fell from 50.0% to 25.0%, contributing to the macro-F1 decline. The loss values use different targets and smoothing than the baseline and should be compared within each run.

For a second exercise 3 attempt, the ensemble gives equal weight to the probabilities of the 128×128 baseline, native 48×48 ResNet-18, and VGGFace2 FaceNet. The three members combine different input processing and pretraining. Averaging can temper confident mistakes when the members disagree. The ensemble reuses the three trained checkpoints and adds no training data. It improved test accuracy by 1.75 points and macro-F1 by 0.97 points over the baseline. Its 87.23% accuracy can be placed alongside the published 95.78% FER-EMFormer result, with the differences in filtering and pretraining described above.

The randomly initialized ResNet-18 reached 67.78% test macro-F1 and 82.49% accuracy. ImageNet initialization improved those scores by 4.54 and 2.99 percentage points under the same 25-epoch training schedule. The initialized model also learned faster. At epoch 5, its validation macro-F1 was 67.7%, versus 48.4% from scratch.

### ResNet-18 versus AlexNet during training

Each row compares both networks at the same epoch. Loss is the weighted, smoothed cross-entropy used in both runs.

| Epoch | Model | Train loss | Val loss | Val accuracy | Val macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ResNet-18 | 2.171 | 1.706 | 67.6% | 46.6% |
| 1 | AlexNet | 2.081 | 1.806 | 63.0% | 40.7% |
| 5 | ResNet-18 | 1.444 | 1.370 | 81.8% | 67.7% |
| 5 | AlexNet | 1.703 | 1.621 | 68.7% | 50.9% |
| 25 | ResNet-18 | 1.047 | 1.281 | 86.5% | 75.1% |
| 25 | AlexNet | 1.297 | 1.352 | 81.5% | 67.5% |

ResNet-18 reduced the loss more quickly and sustained higher validation scores. AlexNet's peak validation macro-F1 of 69.28% at epoch 26 was below ResNet-18's 76.37% at epoch 21. Both validation losses levelled off near the end while training losses continued to fall. On test, AlexNet was 4.08 percentage points behind on macro-F1 and 3.17 points behind on accuracy.

ResNet-18 uses residual connections and batch normalization, which can help fine-tune a deeper hierarchy of features. AlexNet starts with an 11×11 stride-4 convolution and downsamples early, so subtle local features may become harder to retain. Its 57.0 million parameters are concentrated in its large fully connected classifier, versus 11.2 million parameters for the baseline. These are plausible contributors, and the 25- and 30-epoch OneCycle schedules also differ. [ResNet paper](https://arxiv.org/abs/1512.03385) · [AlexNet paper](https://papers.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### Face recognition pretraining

FaceNet reached 73.99% test macro-F1 and 86.49% accuracy. It improved on the baseline by 1.67 and 1.01 percentage points, on random initialization by 6.21 and 4.00 points, and on AlexNet by 5.75 and 4.18 points. Relative to the baseline, disgust F1 rose from 45.0% to 51.3%, and fear F1 rose from 62.1% to 65.1%. Contempt F1 fell from 48.0% to 46.5%.

### Photo inference

The ensemble's 77.24% validation macro-F1 selected it for photo inference. FaceNet achieved the highest test macro-F1. OpenCV detected and cropped a face in each of the three workshop photos. The predicted classes and scores were neutral (73.0%), happy (72.9%), and neutral (75.5%). The public repository includes the inference script and the benchmark results. The workshop photos and visual predictions are kept local.

See the [README](README.md#reproduce-the-workshop) for instructions on running inference with your own photos.

## Included results

- [Comparison CSV](results/comparison.csv) lists the six single-model runs and test differences from baseline in percentage points.
- [Per-experiment data](results/experiments) contains the run settings, epoch histories, test scores, and aggregate plots.
- [ResNet-18 versus AlexNet](results/resnet18_vs_alexnet.png) overlays the four training and validation curves.
- [Ensemble results](results/ensemble.json) and [model selection](results/selected_model.json) record the ensemble evaluation and final validation-based selection.

The original runs used Python 3.12.16, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, facenet-pytorch 2.6.0, and OpenCV 4.14.0 on an RTX 4090.
