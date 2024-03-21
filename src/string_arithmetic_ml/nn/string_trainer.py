import torch.nn as nn
from torch import optim
import string_arithmetic_ml.nn.nn_utility_funcs
import string_arithmetic_ml.nn.string_dataset as data
import string_arithmetic_ml.prep.string_arithmetic_generator as generator
from torch.utils.data import DataLoader
from string_arithmetic_ml.nn.architectures.control.rnn import StringRNN
from string_arithmetic_ml.nn.architectures.control.fcn import SimpleFCN
import string_arithmetic_ml.nn.training_loop as loop
from string_arithmetic_ml.prep.utility import master_dir

# Define dataset and data loader
training_set, testing_set, validation_set = data.split_dataset(
    data.StringDataset.from_json(master_dir('cache/simple_dataset.json')), .75, .15)

batch_size = 2 ** 4

training_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, drop_last=True)
testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Create the model
model = SimpleFCN(batch_size)

# Define a loss function
loss_func = nn.MSELoss()

# Define optimizer
learning_rate = 0.05
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    args = loop.Arguments(
        model,
        loss_func,
        optimizer,
        training_set,
        validation_set,
        training_loader,
        validation_loader,
        200,
        string_arithmetic_ml.nn.nn_utility_funcs.default_model_save_path,
        epochal_update=lambda args, epoch: print(
            f'For the epoch {epoch}, the accuracy was {args.epochal_validation_mean_loss}.'
            f' Best so far was {args.best_validation_loss}')
    )
    loop.loop(args)
