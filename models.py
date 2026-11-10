"""The networks.

- alexnet:   the network used in the AffectNet paper
- resnet18:  a newer, smaller and usually more accurate network (default)
- small_cnn: a compact CNN trained from scratch, to show how much ImageNet pretraining helps

In each case the ImageNet classifier is replaced with a new layer that outputs one score per expression.
"""
import torch
import torch.nn as nn
import torchvision

from augment import IMAGENET_MEAN, IMAGENET_STD


class FaceNetInput(nn.Module):
    def forward(self, x):
        # prepare() repeats grayscale across RGB. Undo its normalization, then
        # apply FaceNet's fixed standardization (pixel - 127.5) / 128.
        gray = x[:, :1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
        return ((gray * 255 - 127.5) / 128).expand(-1, 3, -1, -1)


def build_model(name, num_classes, pretrained=True):
    tvm = torchvision.models
    if name in ("resnet18", "resnet18_native"):
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if name == "resnet18_native":
            stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if pretrained:
                with torch.no_grad():
                    stem.weight.copy_(model.conv1.weight[:, :, 2:5, 2:5])
            model.conv1 = stem
            # The first residual stage sees 24x24, the final one 3x3.
            model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif name == "alexnet":
        model = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "facenet":
        from facenet_pytorch import InceptionResnetV1

        backbone = InceptionResnetV1(
            pretrained="vggface2" if pretrained else None,
            classify=True,
            num_classes=num_classes,
        )
        model = nn.Sequential(FaceNetInput(), backbone)
    elif name == "small_cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        model = nn.Sequential(block(3, 64), block(64, 128), block(128, 256), block(256, 512),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def load_expression_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config, class_names = checkpoint["config"], checkpoint["class_names"]
    model = build_model(config["model"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, class_names
