import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import numpy as np
import torch
from torch.utils.data import Dataset
from string_arithmetic_ml.prep.utility import master_dir


@dataclass
class MathiasDataset(Dataset):
    data: torch.Tensor
    default_save_path: ClassVar[Path] = master_dir('cache/mathias_ver_cache/dataset.pt')

    @classmethod
    def from_size(cls, size: int):
        return cls(torch.rand(size, 2))

    @classmethod
    def from_pt_file(cls, file: str | Path = None):
        file = MathiasDataset.default_save_path if file is None else file
        return cls(torch.load(file))

    def save(self, path: Path = None):
        path = MathiasDataset.default_save_path if path is None else path
        torch.save(self.data, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.data[index].sum(dim=-1, keepdim=True)


@dataclass
class MathiasDatasetAllOperators(Dataset):
    data: torch.Tensor
    default_save_path: ClassVar[Path] = master_dir('cache/mathias_ver_cache/dataset_all_op.pt')

    @classmethod
    def from_size(cls, size: int):
        initial = torch.rand(size, 2)
        initial += 1e-7
        result = np.empty((0, initial.shape[1] + 1),
                          dtype=np.float32)  # + 1 because its the size of the row from intial
        for vector in initial:
            new = np.append(vector, np.float32(random.randint(0, 3)))
            result = np.vstack((result, new))
        return cls(torch.from_numpy(result))

    @classmethod
    def from_pt_file(cls, file: str | Path = None):
        file = MathiasDatasetAllOperators.default_save_path if file is None else file
        return cls(torch.load(file))

    def save(self, path: Path = None):
        path = MathiasDatasetAllOperators.default_save_path if path is None else path
        torch.save(self.data, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        # <editor-fold desc="Operator Specific Label Generation">
        label: torch.Tensor = None
        if item[-1] == 0:
            label = torch.tensor(item[0].item() + item[1].item())
        elif item[-1] == 1:
            label = torch.tensor(item[0].item() - item[1].item())
        elif item[-1] == 2:
            label = torch.tensor(item[0].item() * item[1].item())
        elif item[1] == 0:
            raise ZeroDivisionError(f'Sample {item} had 0 in the divisor')
        elif item[-1] == 3:
            label = torch.tensor(item[0].item() / item[1].item())
        # </editor-fold>
        return self.data[index], label.view(1)


if __name__ == '__main__':
    # MathiasDataset.from_size(10000).save()
    from string_arithmetic_ml.nn.string_dataset import split_dataset

    mode = '_all_op_complex'

    data = MathiasDatasetAllOperators.from_size(500)
    data.save()
    training_set, testing_set, validation_set = split_dataset(
        data, .75, .15)
    torch.save(training_set, master_dir(f'cache/mathias_ver_cache/training{mode}.pt'))
    torch.save(testing_set, master_dir(f'cache/mathias_ver_cache/testing{mode}.pt'))
    torch.save(validation_set, master_dir(f'cache/mathias_ver_cache/validation{mode}.pt'))
    print(data[0])
