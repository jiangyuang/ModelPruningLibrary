import torch
import torchvision.models
from typing import Any

from .base_model import BaseModel


class SqueezeNet(BaseModel):
    def __init__(self, model: torchvision.models.SqueezeNet):
        super(SqueezeNet, self).__init__()
        self.clone_from_model(model)
        self.process_layers()

    def process_layers(self):
        self.collect_prunable_layers()
        self.convert_eligible_layers()
        self.collect_prunable_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    return SqueezeNet(torchvision.models.squeezenet1_0(pretrained, progress, **kwargs))


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    return SqueezeNet(torchvision.models.squeezenet1_1(pretrained, progress, **kwargs))
