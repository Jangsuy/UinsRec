
import torch
import torch.distributed as dist

from config.global_config import GlobalCONFIG as G


def print_rank_0(str):
    try:
        if torch.distributed.get_rank() == 0:
            print(str)
    except:
        print(str)
    return None


def gpu_allocated():
    return torch.cuda.max_memory_allocated() / 1073741824.0  # Giga bytes


def tensor(x, dtype=torch.float32, copy=False):
    if isinstance(x, torch.Tensor):
        return x
    if copy:
        return torch.tensor(x, device=G.DEVICE, dtype=dtype)
    else:
        return torch.as_tensor(x, device=G.DEVICE, dtype=dtype)


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    else:
        return t


def get_model_memory(model):
    return sum([p.element_size() * p.nelement() for p in model.parameters()]) / (1024 ** 3)


def get_model_parameters(model):
    return sum([p.nelement() for p in model.parameters()])


def get_trainable_model_parameters(model):
    trainable_weights = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_weights += param.nelement()
    return trainable_weights


def zeros(size, dtype=None, grads=False):
    return torch.zeros(size=size, dtype=dtype, device=G.DEVICE, requires_grad=grads)


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True)
            for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


grad_all_gather = AllGather.apply
