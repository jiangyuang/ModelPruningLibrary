from enum import Enum


class NAMES(Enum):
    MNIST = 0
    FashionMNIST = 1
    FEMNIST = 2
    CIFAR10 = 3


class MNIST:
    N_FEATURES = 28 * 28
    N_CLASSES = 10
    N_TRAIN = 60000
    N_TEST = 10000


class FashionMNIST:
    N_FEATURES = 28 * 28
    N_CLASSES = 10
    N_TRAIN = 60000
    N_TEST = 10000


class CIFAR10:
    N_FEATURES = 3 * 32 * 32
    N_CLASSES = 10
    N_TRAIN = 50000
    N_TEST = 10000


class FEMNIST:
    N_FEATURES = 28 * 28
    N_CLASSES = 62
    N_TOTAL = 35948
    N_TRAIN = 35559
    N_SAMPLES = 389
    N_TEST = 4090
    N_USERS = 193


def get_dataset_name(name: str):
    if name.lower() == "mnist":
        return NAMES.MNIST
    elif name.lower() in ["fashion mnist", "fashionmnist", "fashion_mnist", "fashion-mnist"]:
        return NAMES.FashionMNIST
    elif name.lower() == "femnist":
        return NAMES.FEMNIST
    else:
        raise ValueError("{} is not supported.".format(name))


def get_dataset_constants(name: str):
    dataset_name = get_dataset_name(name)
    if dataset_name == NAMES.MNIST:
        return MNIST
    elif dataset_name == NAMES.FashionMNIST:
        return FashionMNIST
    elif dataset_name == NAMES.FEMNIST:
        return FEMNIST
    else:
        raise ValueError("{} is not supported.".format(name))
