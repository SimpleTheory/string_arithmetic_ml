import torch
import torch.nn as nn
from string_arithmetic_ml.nn.nn_utility_funcs import cuda


class StringRNN(nn.Module):

    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(StringRNN, self).__init__()
        cuda(self)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True, dtype=torch.int)
        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.int)

    def init_hidden(self):
        # Could be initialized in a variety of ways
        return torch.zeros(1, self.hidden_size)

    def forward(self, inputs: torch.tensor, hidden_state: torch.tensor = None):
        if not hidden_state:
            hidden_state = self.init_hidden()
        output, hidden = self.rnn(inputs, hidden_state)
        output = self.fc(output)
        return output
