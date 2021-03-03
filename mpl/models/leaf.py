from torch import nn as nn

from .base_model import BaseModel
from ..nn.conv2d import DenseConv2d
from ..nn.linear import DenseLinear

__all__ = ["Conv2", "conv2", "Conv4", "conv4"]


class Conv2(BaseModel):
    def __init__(self):
        super(Conv2, self).__init__()
        self.features = nn.Sequential(DenseConv2d(1, 32, kernel_size=5, padding=2),  # 32x28x28
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, stride=2),  # 32x14x14
                                      DenseConv2d(32, 64, kernel_size=5, padding=2),  # 64x14x14
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, stride=2))  # 64x7x7

        self.classifier = nn.Sequential(DenseLinear(64 * 7 * 7, 2048),
                                        nn.ReLU(inplace=True),
                                        DenseLinear(2048, 62))
        self.collect_prunable_layers()

    def forward(self, inp):
        out = self.features(inp)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Conv4(BaseModel):
    def __init__(self):
        super(Conv4, self).__init__()
        self.features = nn.Sequential(DenseConv2d(3, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.MaxPool2d(2),
                                      DenseConv2d(32, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.MaxPool2d(2),
                                      DenseConv2d(32, 32, kernel_size=3, padding=2),
                                      nn.BatchNorm2d(32),
                                      nn.MaxPool2d(2),
                                      DenseConv2d(32, 32, kernel_size=3, padding=2),
                                      nn.BatchNorm2d(32),
                                      nn.MaxPool2d(2))

        self.classifier = DenseLinear(in_features=32 * 6 * 6, out_features=2)

    def forward(self, inp):
        out = self.features(inp)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def conv2() -> Conv2:
    return Conv2()


def conv4() -> Conv4:
    return Conv4()

# TODO: define pretrain etc.
