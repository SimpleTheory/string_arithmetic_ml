import torch
import torch.nn as nn
from string_arithmetic_ml.nn.nn_utility_funcs import cuda
from string_arithmetic_ml.prep.string_arithmetic_generator import simple_max_unit_length


class SimpleFCN(nn.Module):
    def __init__(
            self,
            batch_size: int,
            input_size=simple_max_unit_length,
            hidden_size=simple_max_unit_length // 2,
            output_size=1
    ):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*batch_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        cuda(self)

    def forward(self, input_tensor):
        activated_hidden_tensor = torch.relu(self.fc1(input_tensor))
        result = self.fc2(activated_hidden_tensor)
        return result

