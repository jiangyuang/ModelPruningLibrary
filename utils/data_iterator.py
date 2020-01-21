import math
import torch


class DataIterator:
    def __init__(self, batch_size, data_loader=None, data: tuple = None, device=None,
                 perm=None):
        if (data_loader is None) == (data is None):
            raise ValueError("Need either data loader or data")

        if data_loader is not None:
            inputs_data, labels_data = next(iter(data_loader))
        else:
            inputs_data, labels_data = data

        self.inputs = inputs_data.float()
        self.labels = labels_data.float()
        if perm is not None:
            if not isinstance(perm, torch.Tensor):
                perm = torch.tensor(perm)
            self.inputs = self.inputs[perm]
            self.labels = self.labels[perm]
        if device is not None:
            self.inputs = self.inputs.to(device)
            self.labels = self.labels.to(device)
        self.size = self.labels.size(0)
        self.batch_size = batch_size
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset == self.size:
            self.offset = 0
            raise StopIteration
        else:
            start = self.offset
            end = start + self.batch_size
            if end > self.size:
                end = self.size
            self.offset = end
            return self.inputs[start:end], self.labels[start:end]

    def get_next_batch(self):
        try:
            return self.__next__()
        except StopIteration:
            return self.__next__()

    def __len__(self):
        return math.ceil(self.size / self.batch_size)
