from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset
from string_arithmetic_ml.prep.utility import master_dir

default_save_path_model = master_dir('cache/mathias_ver_cache/dataset.pt')

@dataclass
class MathiasDataset(Dataset):
    data: torch.Tensor

    @classmethod
    def from_size(cls, size: int):
        return cls(torch.rand(size, 2))

    @classmethod
    def from_pt_file(cls, file: str | Path = default_save_path_model):
        return cls(torch.load(file))

    def save(self, path: Path = default_save_path_model):
        torch.save(self.data, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.data[index].sum(dim=-1, keepdim=True)


if __name__ == '__main__':
    # MathiasDataset.from_size(10000).save()
    from string_arithmetic_ml.nn.string_dataset import split_dataset
    a = MathiasDataset.from_pt_file()
    data = a
    training_set, testing_set, validation_set = split_dataset(
        a, .75, .15)
    torch.save(training_set, master_dir('cache/mathias_ver_cache/training.pt'))
    torch.save(testing_set, master_dir('cache/mathias_ver_cache/testing.pt'))
    torch.save(validation_set, master_dir('cache/mathias_ver_cache/validation.pt'))
    print(a[0])