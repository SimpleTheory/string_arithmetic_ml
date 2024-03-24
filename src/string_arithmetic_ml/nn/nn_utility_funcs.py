import functools
import torch

def no_grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
            return result

    return wrapper


def cuda(obj):
    return obj.cuda() if torch.cuda.is_available() else obj


def batch_cuda(*args):
    return [cuda(arg) for arg in args]
