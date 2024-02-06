import torch
import torch.nn as nn
from string_arithmetic_ml.nn.nn_utility_funcs import cuda


class StringRNN(nn.Module):

    def __init__(self, input_size=2, hidden_size=1, output_size=1):
        super(StringRNN, self).__init__()
        cuda(self)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # bidirectional=True, dtype=? int no work sad
        # apparently the output size is 32 118 256, but why
        # self.fc = nn.Linear(hidden_size, 1)  # dtype=? int no work sad

    # def init_hidden(self, batch_size):
    #     # Could be initialized in a variety of ways
    #     return cuda(torch.randn(1, batch_size, self.hidden_size))
    #     # 32 represents the batch size but I don't know how to abstract that here atm

    def forward(self, inputs: torch.tensor):
        # if not hidden_state:
        #     hidden_state = self.init_hidden(inputs.size(0))
        output, hidden = self.rnn(inputs)
        # temp = self.fc(output)
        final = [out[-1] for out in output]
        return final
