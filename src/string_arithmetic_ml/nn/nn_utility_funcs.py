import functools
import torch
from string_arithmetic_ml.prep.utility import master_dir

default_model_save_path = master_dir('cache/model.pth')


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
