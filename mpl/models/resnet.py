import torch
from torch import Tensor
from torch import nn
import torchvision.models
from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck
from typing import Type, Any, Union

from .base_model import BaseModel

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


class ResNet(BaseModel):
    def __init__(self, model: torchvision.models.ResNet):
        super(ResNet, self).__init__()
        self.clone_from_model(model)
        self.process_layers()

    def process_layers(self):
        self.collect_prunable_layers()
        self.convert_eligible_layers()
        self.collect_prunable_layers()

    def collect_prunable_layers(self) -> None:
        """
        removed transition layers from prunable layers
        """
        super(ResNet, self).collect_prunable_layers()
        keep_indices = []
        for layer_idx, name in enumerate(self.prunable_layer_prefixes):
            if "downsample" not in name:
                keep_indices.append(layer_idx)

        self.prunable_layer_prefixes = [self.prunable_layer_prefixes[idx] for idx in keep_indices]
        self.prunable_layers = [self.prunable_layers[idx] for idx in keep_indices]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnet18(pretrained, progress, **kwargs))


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnet34(pretrained, progress, **kwargs))


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnet50(pretrained, progress, **kwargs))


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnet101(pretrained, progress, **kwargs))


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnet152(pretrained, progress, **kwargs))


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnext50_32x4d(pretrained, progress, **kwargs))


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.resnext101_32x8d(pretrained, progress, **kwargs))


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.wide_resnet50_2(pretrained, progress, **kwargs))


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return ResNet(torchvision.models.wide_resnet101_2(pretrained, progress, **kwargs))
