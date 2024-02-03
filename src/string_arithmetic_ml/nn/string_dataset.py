import torch
from torch.utils.data import Dataset
from typing import Callable, Iterable
from dataclasses import dataclass
import string_arithmetic_ml.prep.string_arithmetic_generator as generator
from string_arithmetic_ml.nn.nn_utility_funcs import cuda
from pathlib import Path


@dataclass
class StringDataset(Dataset):
    data: list[tuple[str, int]] = None  # Should be (arithmetic string, solution)

    @classmethod
    def from_size(cls, size: int):
        return cls(generator.generate_dataset(size))

    @classmethod
    def from_json(cls, json: str | Path = generator.default_save_path):
        return cls(generator.load(json))

    def save(self, path=generator.default_save_path):
        generator.save(self.data, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int, encoding_scheme: Callable[[str], Iterable[tuple[int, int]]] = generator.encoder):
        """
        Logic to process and return a single sample.
        """
        sample: torch.Tensor = torch.tensor(encoding_scheme(self.data[index][0]), dtype=torch.int)
        solution: int = self.data[index][1]
        return sample, solution

def split_dataset(dataset, training: float = .75, validation: float = .15):
    testing = 1 - (training + validation)
    return torch.utils.data.random_split(dataset, [training, validation, testing])
