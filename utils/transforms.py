import torch
from torch.nn.functional import one_hot


class Flatten:
    def __call__(self, img: torch.FloatTensor):
        return img.reshape((-1))


class OneHot:
    def __init__(self, n_classes, to_float: bool = False):
        self.n_classes = n_classes
        self.to_float = to_float

    def __call__(self, label):
        label = label.clone() if isinstance(label, torch.Tensor) else torch.tensor(label)
        return one_hot(label, self.n_classes).float() if self.to_float else one_hot(label, self.n_classes)
