import torch
import torchvision.models
from typing import Any

from .base_model import BaseModel

__all__ = ['AlexNet', 'alexnet']


class AlexNet(BaseModel):
    def __init__(self, model: torchvision.models.AlexNet):
        super(AlexNet, self).__init__()
        self.clone_from_model(model)
        self.process_layers()

    def process_layers(self):
        self.collect_prunable_layers()
        self.convert_eligible_layers()
        self.collect_prunable_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    return AlexNet(torchvision.models.alexnet(pretrained, progress, **kwargs))
