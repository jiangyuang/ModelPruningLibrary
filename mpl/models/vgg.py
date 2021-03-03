import torch
import torch.nn as nn
import torchvision.models
from typing import Any

from .base_model import BaseModel

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(BaseModel):
    def __init__(self, model: torchvision.models.VGG):
        super(VGG, self).__init__()
        self.clone_from_model(model)
        self.process_layers()

    def process_layers(self):
        self.collect_prunable_layers()
        self.convert_eligible_layers()
        self.collect_prunable_layers()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg11(pretrained, progress, **kwargs))


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg11_bn(pretrained, progress, **kwargs))


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg13(pretrained, progress, **kwargs))


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg13_bn(pretrained, progress, **kwargs))


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg16(pretrained, progress, **kwargs))


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg16_bn(pretrained, progress, **kwargs))


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg19(pretrained, progress, **kwargs))


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return VGG(torchvision.models.vgg19_bn(pretrained, progress, **kwargs))
