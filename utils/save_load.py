import pickle
import copyreg
import io
import os
import torch

bytes_types = (bytes, bytearray)


def _get_int_type(max_val: int):
    assert max_val >= 0
    max_uint8 = 1 << 8
    max_int16 = 1 << 15
    max_int32 = 1 << 31
    if max_val < max_uint8:
        return torch.uint8
    elif max_val < max_int16:
        return torch.int16
    elif max_val < max_int32:
        return torch.int32
    else:
        return torch.int64


def _sparse_tensor_constructor(indices, values, size):
    return torch.sparse.FloatTensor(indices.to(torch.long), values, size).coalesce()


def _reduce(x: torch.sparse.FloatTensor):
    # dispatch table cannot distinguish between torch.sparse.FloatTensor and torch.Tensor
    if isinstance(x, torch.sparse.FloatTensor) or isinstance(x, torch.sparse.LongTensor):
        int_type = _get_int_type(torch.max(x._indices()).item())
        return _sparse_tensor_constructor, (x._indices().to(int_type), x._values(), x.size())
    else:
        return torch.Tensor.__reduce_ex__(x, pickle.HIGHEST_PROTOCOL)  # use your own protocol


class ExtendedPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[torch.Tensor] = _reduce
    # tried to use torch.sparse.FloatTensor instead of torch.Tensor but did not work (?)


def dumps(obj):
    f = io.BytesIO()
    ExtendedPickler(f).dump(obj)
    res = f.getvalue()
    assert isinstance(res, bytes_types)
    return res


def loads(res):
    return pickle.loads(res)


def save(obj, f):
    with open(f, "wb") as opened_f:
        ExtendedPickler(opened_f).dump(obj)


def mkdir_save(obj, f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    save(obj, f)


def load(f):
    with open(f, 'rb') as opened_f:
        return pickle.load(opened_f)
