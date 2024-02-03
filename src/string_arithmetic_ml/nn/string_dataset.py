import torch
from torch.utils.data import Dataset
from typing import Callable, Iterable
from dataclasses import dataclass
import string_arithmetic_ml.prep.string_arithmetic_generator as generator
from string_arithmetic_ml.nn.nn_utility_funcs import cuda
from pathlib import Path


@dataclass
class StringDataset(Dataset):
    data: list[generator.Sample] = None  # Should be (arithmetic string, solution)

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
        Logic to process and return a single sample. Make sure that the inputs of the first layer and your final output
        of your model line up with the correct sizes
        (size listed in encoding scheme used default is `generator.max_unit_length`, size of answer should be a scalar int)
        and dtype (torch.int, int) of the Dataset.
        """
        sample: torch.Tensor = torch.tensor(encoding_scheme(self.data[index].problem), dtype=torch.int)
        solution: int = self.data[index].solution
        return sample, solution

def split_dataset(dataset, training: float = .75, validation: float = .15):
    testing = 1 - (training + validation)
    return torch.utils.data.random_split(dataset, [training, validation, testing])
