import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from string_arithmetic_ml.mathias_version.dataset import MathiasDataset
# from string_arithmetic_ml.nn.string_dataset import split_dataset
from string_arithmetic_ml.prep.utility import master_dir
import string_arithmetic_ml.nn.training_loop as loop
from string_arithmetic_ml.nn.model_test_loop import ModelLossTest


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


# Create the model
model = Net()
model_save_path = master_dir('cache/mathias_ver_cache/model.pth')

# Dataset
# data = MathiasDataset.from_size(10000)
# training_set, testing_set, validation_set = split_dataset(data, .75, .15)

data = MathiasDataset.from_pt_file()
training_set = torch.load(master_dir('cache/mathias_ver_cache/training.pt'))
validation_set = torch.load(master_dir('cache/mathias_ver_cache/validation.pt'))
testing_set = torch.load(master_dir('cache/mathias_ver_cache/testing.pt'))

batch_size = 2 ** 4

# training_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
# testing_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
# validation_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, drop_last=True)


# Define loss function and optimizer
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    def f(arg, epoch):
        if epoch % 500 == 0:
            print(
                f'For the epoch {epoch}, the err was {arg.epochal_validation_mean_loss}.'
                f' Best so far was {arg.best_validation_loss}')
    args = loop.Arguments(
        model,
        loss_func,
        optimizer,
        training_set,
        validation_set,
        training_loader,
        validation_loader,
        6000,
        model_save_path,
        epochal_update=f,
        from_existing_model=True
    )
    # loop.loop(args)
    model.load_state_dict(torch.load(model_save_path))
    evaluation = ModelLossTest.from_training_args(args, testing_set, testing_loader, model=model)
    evaluation.save_sample_list(master_dir('cache/mathias_ver_cache/evaluation_list.pt'))
    print(evaluation.mean_loss)
    example = model(torch.Tensor([.5, .2]))
    print(example)
    print(.7-example.item())
    print(model.fc.weight)
    # a = ModelLossTest.load_sample_list_from_file(master_dir('cache/mathias_ver_cache/evaluation_list.pt'))
    # print(a[:5])

