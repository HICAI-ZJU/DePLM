import torch
from torch import Tensor
import numpy as np

def _find_repeats(data: Tensor) -> Tensor:
    """Find and return values which have repeats i.e. the same value are more than once in the tensor."""
    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]

def _rank_data(data: Tensor) -> Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1). Duplicates of the
    same value will be assigned the mean of their rank.

    Adopted from `Rank of element tensor`_

    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank

def quick_sort(arr):
    partial_sorted_arr = []
    if len(arr) < 2: return arr
    stack = []
    stack.append([0, len(arr)-1])

    while stack:
        tmp_stack = []
        while stack:
            l, r = stack.pop()
            index = partition(arr, l, r)
            if l < index - 1:
                tmp_stack.append([l, index-1])
            if r > index + 1:
                tmp_stack.append([index+1, r])
        stack = tmp_stack
        partial_sorted_arr.append(arr.copy())
    return partial_sorted_arr


def partition(arr, s, t):
    pivot = np.random.randint(s, t+1)
    arr[s], arr[pivot] = arr[pivot], arr[s]
    tmp = arr[s]
    while s < t:
        while s < t and arr[t] >= tmp: t -= 1
        arr[s] = arr[t]
        while s < t and arr[s] <= tmp: s += 1
        arr[t] = arr[s]
    arr[s] = tmp
    return s