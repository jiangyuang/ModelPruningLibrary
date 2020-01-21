import torch
import math


def random_split(length, n_subsets) -> list:
    size = math.ceil(length / n_subsets)

    list_offsets = [0] + [size * (idx + 1) for idx in range(n_subsets - 1)] + [length]

    indices = torch.randperm(length).tolist()
    split = [indices[list_offsets[idx]:list_offsets[idx + 1]] for idx in range(n_subsets)]
    return split


def rearrange_split(random_indices: list, shift):
    for i in range(1, len(random_indices)):
        assert len(random_indices[i]) == len(random_indices[0])

    perm = []
    offset = 0
    while True:
        for subset in random_indices:
            start = offset
            end = offset + shift
            if end > len(subset):
                end = len(subset)
            perm.extend(subset[start:end])
        offset += shift
        if len(perm) == sum([len(subset) for subset in random_indices]):
            break
    return perm
