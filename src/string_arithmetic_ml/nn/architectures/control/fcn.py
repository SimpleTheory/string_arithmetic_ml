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
        self.batch_size = batch_size
        self.fc1 = nn.Linear(14, hidden_size*batch_size) # 14x48
        self.fc2 = nn.Linear(48, 5) # 48x5
        self.fc3 = nn.Linear(5, output_size) # 5x1
        cuda(self)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(self.batch_size, -1)
        activated_hidden_tensor = torch.relu(self.fc1(input_tensor))
        activated_hidden_tensor2 = torch.relu(self.fc2(activated_hidden_tensor))
        result = self.fc3(activated_hidden_tensor2)
        return result.squeeze(-1)



"""
16-7-2 x 2x48

16-7-48 x 48x1

"""