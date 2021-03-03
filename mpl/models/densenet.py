import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.models
from typing import Any

from .base_model import BaseModel

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


class DenseNet(BaseModel):
    def __init__(self, model: torchvision.models.DenseNet):
        super(DenseNet, self).__init__()
        self.clone_from_model(model)
        self.process_layers()

    def collect_prunable_layers(self) -> None:
        """
        removed transition layers from prunable layers
        """
        super(DenseNet, self).collect_prunable_layers()
        keep_indices = []
        for layer_idx, name in enumerate(self.prunable_layer_prefixes):
            if "transition" not in name:
                keep_indices.append(layer_idx)

        self.prunable_layer_prefixes = [self.prunable_layer_prefixes[idx] for idx in keep_indices]
        self.prunable_layers = [self.prunable_layers[idx] for idx in keep_indices]

    def process_layers(self):
        self.collect_prunable_layers()
        self.convert_eligible_layers()
        self.collect_prunable_layers()

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return DenseNet(torchvision.models.densenet121(pretrained, progress, **kwargs))


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return DenseNet(torchvision.models.densenet161(pretrained, progress, **kwargs))


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return DenseNet(torchvision.models.densenet169(pretrained, progress, **kwargs))


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return DenseNet(torchvision.models.densenet201(pretrained, progress, **kwargs))
