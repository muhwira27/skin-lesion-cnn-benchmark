from __future__ import annotations

import torch.nn as nn
import timm
import torchvision.models as tvm

TORCHVISION_MODELS = {
    "resnet18",
    "resnet50",
    "densenet121",
    "resnext50_32x4d",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
}


def _try_torchvision_weights(name: str):
    try:
        weights_enum_name = {
            "resnet18": "ResNet18_Weights",
            "resnet50": "ResNet50_Weights",
            "densenet121": "DenseNet121_Weights",
            "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
            "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights",
        }[name]
        weights_enum = getattr(tvm, weights_enum_name)
        return weights_enum.DEFAULT
    except Exception:
        return None


def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> None:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return

        if isinstance(model.classifier, nn.Sequential):
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    in_features = model.classifier[i].in_features
                    model.classifier[i] = nn.Linear(in_features, num_classes)
                    return

    raise RuntimeError(f"Could not replace classifier for model: {name}")


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone in TORCHVISION_MODELS:
        weights = _try_torchvision_weights(backbone)
        ctor = getattr(tvm, backbone)
        if weights is not None:
            model = ctor(weights=weights)
        else:
            # fallback
            model = ctor()
        _replace_classifier(model, backbone, num_classes)
        return model

    return timm.create_model(backbone, pretrained=True, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_head(model: nn.Module) -> None:
    # best-effort name heuristic across torchvision + timm
    for name, p in model.named_parameters():
        if any(k in name for k in ("fc", "classifier", "head")):
            p.requires_grad = True
