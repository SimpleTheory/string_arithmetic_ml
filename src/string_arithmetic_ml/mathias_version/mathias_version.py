import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import string_arithmetic_ml.mathias_version.dataset as datasets
from string_arithmetic_ml.mathias_version.dataset import MathiasDatasetAllOperators
from string_arithmetic_ml.nn.nn_utility_funcs import cuda
# from string_arithmetic_ml.nn.string_dataset import split_dataset
from string_arithmetic_ml.prep.utility import master_dir
import string_arithmetic_ml.nn.training_loop as loop
from string_arithmetic_ml.nn.model_test_loop import ModelLossTest
import string_arithmetic_ml.mathias_version.nns as nets


# Define the model
mode = '_division'

# Create the model
model = nets.AllOpNetComplex()
model_save_path = master_dir(f'cache/mathias_ver_cache/model{mode}.pth')

# Dataset
# data = MathiasDataset.from_size(10000)
# training_set, testing_set, validation_set = split_dataset(data, .75, .15)

data = datasets.MathiasDatasetAllOperators.from_pt_file()
training_set = torch.load(master_dir(f'cache/mathias_ver_cache/training{mode}.pt'))
validation_set = torch.load(master_dir(f'cache/mathias_ver_cache/validation{mode}.pt'))
testing_set = torch.load(master_dir(f'cache/mathias_ver_cache/testing{mode}.pt'))

batch_size = 2 ** 6

# training_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
# testing_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
# validation_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, drop_last=True)


# Define loss function and optimizer
loss_func = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    def f(arg: loop.Arguments, epoch: int):
        if epoch == 0:
            arg.kwargs['last_best_loss'] = args.best_validation_loss
            arg.kwargs['last_best_loss_epoch'] = epoch

        elif arg.best_validation_loss != arg.kwargs['last_best_loss']:
            arg.kwargs['last_best_loss'] = args.best_validation_loss
            arg.kwargs['last_best_loss_epoch'] = epoch

        elif (epoch - arg.kwargs['last_best_loss_epoch']) > 300:
            arg.schedulers[0].step()
            print('STEP!')

        if epoch % 10 == 0:
            print(
                f'For the epoch {epoch}, the err was {arg.epochal_validation_mean_loss}.'
                f' Best so far was {arg.best_validation_loss}. Last best was achieved {epoch - arg.kwargs['last_best_loss_epoch']} epochs ago.')

    args = loop.Arguments(
        model,
        loss_func,
        optimizer,
        training_set,
        validation_set,
        training_loader,
        validation_loader,
        10000,
        model_save_path,
        schedulers=[torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)],
        epochal_update=f,
        from_existing_model=False

    )
    # loop.loop(args)
    model.load_state_dict(torch.load(model_save_path))
    evaluation = ModelLossTest.from_training_args(args, testing_set, testing_loader, model=model)
    evaluation.save_sample_list(master_dir(f'cache/mathias_ver_cache/evaluation_list{mode}.pt'))
    # print(evaluation.mean_loss)
    example = model(cuda(torch.Tensor([-.1, -.5, np.float32(3)])))
    actual = -.1 / -.5
    print(f'example: {example.item()} | actual: {actual}')
    print('err in sample', abs(actual-example.item()))
    # a = ModelLossTest.load_sample_list_from_file(master_dir('cache/mathias_ver_cache/evaluation_list.pt'))
    # print(a[:5])

