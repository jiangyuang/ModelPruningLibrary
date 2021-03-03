from torch import nn as nn
from .base_model import BaseModel

from ..nn.linear import DenseLinear

__all__ = ["LeNet5", "lenet5"]


class LeNet5(BaseModel):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.classifier = nn.Sequential(DenseLinear(784, 300),
                                        nn.ReLU(inplace=True),
                                        DenseLinear(300, 100),
                                        nn.ReLU(inplace=True),
                                        DenseLinear(100, 10))

        self.collect_prunable_layers()

    def forward(self, inputs):
        return self.classifier(inputs)


def lenet5() -> LeNet5:
    return LeNet5()
