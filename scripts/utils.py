import torch
from typing import Callable, Tuple, Dict


def set_cuda_device(cuda_device, verbose=True):
    available_cuda_device_num = torch.cuda.device_count()
    available_cuda_device_ids = list(range(available_cuda_device_num)) + [-1]

    if available_cuda_device_num == 0 and verbose:
        print("\033[31m[WARN] No CUDA Device Found on This Machine.\n \033[0m")

    assert isinstance(cuda_device, int) and cuda_device in available_cuda_device_ids, \
        f"Error: Wrong CUDA Device Value Encountered! It Should in {available_cuda_device_ids}\n"

    if cuda_device >= 0 and cuda_device < available_cuda_device_num:
        device = torch.device("cuda:" + str(cuda_device))
        torch.cuda.set_device(device)
        cuda_device_id = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_device_id)

        if verbose:
            print("\033[31m[INFO] Device ID: %s \033[0m" % device)
            print("\033[31m[INFO] Device Name: %s\n\n \033[0m" % cuda_device_name)

    else:
        device = torch.device("cpu")
        if verbose:
            print("\033[31m[INFO] Device ID: %s \n\n\033[0m" % device)

    return device


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix).squeeze(1)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size()[i + 1])
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()



StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]

