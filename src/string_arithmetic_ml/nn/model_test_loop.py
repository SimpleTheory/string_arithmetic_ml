# TODO
# Make similiar args class and create something similiar to validate but with only one pass through
# Also create a factory from args to this
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import torch.utils.data
from typing import SupportsFloat as Numeric
from string_arithmetic_ml.nn.training_loop import data_iterator, Arguments
from string_arithmetic_ml.nn.nn_utility_funcs import batch_cuda, cuda, no_grad


@dataclass
class Sample:
    input: torch.Tensor = None
    label: torch.Tensor = None
    output: torch.Tensor = None
    loss: Numeric = None

    def combine(self):
        return [self.input, self.label, self.output, self.loss]

    @classmethod
    def from_tensor_list(cls, tensor_list: list[torch.Tensor]):
        return Sample(*tensor_list)

@dataclass
class ModelLossTest:
    model: torch.nn.Module
    loss_function: 'Loss Object'
    test_set: torch.utils.data.dataset.Subset
    test_loader: torch.utils.data.dataloader.DataLoader
    sample_list: list[Sample] = field(default_factory=lambda: [])

    # _number_of_batches_evaluated: int = 0

    @classmethod
    def from_training_args(cls, args: Arguments, test_set, test_loader, model=None):
        return cls(args.model if model is None else model, args.loss_function, test_set, test_loader)

    def __post_init__(self):
        # self.model, self.loss_function = batch_cuda(self.model, self.loss_function)
        self.model = cuda(self.model)
        self.loop_for_test()

    def loss_values(self):
        for sample in self.sample_list:
            yield sample.loss

    @no_grad
    def loop_for_test(self):
        self.model.eval()
        for problems, labels in data_iterator(self.test_loader):
            outputs = cuda(self.model(problems))
            loss = self.loss_function(outputs, labels)
            self.sample_list.append(
                Sample(
                    problems, labels, outputs, loss
                )
            )

    @property
    def mean_loss(self):
        return sum(self.loss_values()) / len(self.sample_list)

    def save_sample_list(self, path):
        torch.save([i.combine() for i in self.sample_list], path)

    @staticmethod
    def load_sample_list_from_file(path: Path):
        return [Sample.from_tensor_list(tensor_list) for tensor_list in torch.load(path)]
