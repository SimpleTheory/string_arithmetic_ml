import torch
from torch import nn
from string_arithmetic_ml.nn.nn_utility_funcs import cuda


class SimpleAdditionNet(nn.Module):
    def __init__(self):
        super(SimpleAdditionNet, self).__init__()
        self.fc = nn.Linear(2, 1)
        cuda(self)

    def forward(self, x):
        x = self.fc(x)
        return x

class AllOpNet(nn.Module):
    def __init__(self):
        super(AllOpNet, self).__init__()
        self.fc1 = nn.Linear(3, 1)
        # self.fc1 = nn.Linear(3, 12)
        # self.fc2 = nn.Linear(12, 4)
        # self.fc3 = nn.Linear(4, 1)
        cuda(self)

    def forward(self, input):
        result = self.fc1(input)
        # hidden = torch.tanh(self.fc1(input))
        # hidden = torch.tanh(self.fc2(hidden))
        # result = torch.tanh(self.fc3(hidden))
        return result

class AllOpNetComplex(nn.Module):
    def __init__(self):
        super(AllOpNetComplex, self).__init__()
        # self.fc1 = nn.Linear(3, 1)
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 1)
        cuda(self)

    def forward(self, input):
        # result = self.fc1(input)
        hidden = torch.tanh(self.fc1(input))
        hidden = torch.tanh(self.fc2(hidden))
        hidden = torch.tanh(self.fc3(hidden))
        hidden = torch.tanh(self.fc4(hidden))
        hidden = torch.tanh(self.fc5(hidden))
        result = self.fc6(hidden)
        return result
