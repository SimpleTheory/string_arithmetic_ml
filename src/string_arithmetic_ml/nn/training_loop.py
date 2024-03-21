from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import torch.utils.data
from string_arithmetic_ml.nn.nn_utility_funcs import batch_cuda, cuda, no_grad


def update_schedulers(args: 'Arguments', epoch: int):
    for scheduler in args.schedulers:
        scheduler.step()

@dataclass
class Arguments:
    model: torch.nn.Module
    # Either a common loss function from torch.nn.modules.loss or a custom one cf https://saturncloud.io/blog/custom-loss-function-in-pytorch-a-comprehensive-guide/
    loss_function: 'Loss Object'
    optimizer: torch.optim.Optimizer
    training_set: torch.utils.data.dataset.Subset
    validation_set: torch.utils.data.dataset.Subset
    training_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    max_epochs: int
    save_path: Path
    # Function that uses this arguments class and the current epoch to do whatever you want:
    #   if you have schedulers you must step them in the epochal update
    #   any other argument parameters that you want to update you may do so if you would like
    #   or if you would like a read-out per epoch you can put that here
    epochal_update: Callable[['Arguments', int], None] = update_schedulers
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])
    best_validation_loss = 0
    kwargs: dict = field(default_factory=dict)

    @property
    def epochal_validation_mean_loss(self):
        return (self._epochal_validation_loss / self._epochal_num_of_batches_evaluated) \
            if self._epochal_num_of_batches_evaluated > 0 else 0

    def update_epochal_validation_loss(self, batch_loss):
        self._epochal_num_of_batches_evaluated += 1
        self._epochal_validation_loss += batch_loss.item()

    def __post_init__(self):
        self._epochal_validation_loss = 0
        self._epochal_num_of_batches_evaluated = 0
        self.model, self.optimizer, self.loss_function, self.schedulers = \
            batch_cuda(self.model, self.optimizer, self.loss_function, self.schedulers)

    def start_training(self):
        self.model.train()
        self.optimizer.zero_grad()

    # noinspection PyAttributeOutsideInit
    def start_evaluating(self):
        self.model.eval()
        self._epochal_validation_loss = 0
        self._epochal_num_of_batches_evaluated = 0


def data_iterator(dataloader: torch.utils.data.dataloader.DataLoader):
    for batch, sample in enumerate(dataloader):
        yield batch_cuda(*sample)


# <editor-fold desc="Loop Sub-functions">
def train(args: Arguments, epoch):
    args.start_training()
    for problems, labels in data_iterator(args.training_loader):
        outputs = cuda(args.model(problems))
        loss = args.loss_function(outputs, labels)
        loss.backward()
        args.optimizer.step()  # Maybe make the above a closure?


@no_grad
def validate(args: Arguments, epoch):
    args.start_evaluating()
    for problems, labels in data_iterator(args.validation_loader):
        outputs = cuda(args.model(problems))
        loss = args.loss_function(outputs, labels)
        args.update_epochal_validation_loss(loss)

def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def save_logic(args: Arguments, epoch):
    if args.epochal_validation_mean_loss < args.best_validation_loss:
        args.best_validation_loss = args.epochal_validation_mean_loss
        save_model(args.model, args.save_path)
# </editor-fold>

def loop(args: Arguments):
    for epoch in range(args.max_epochs):
        train(args, epoch)
        print('train done')
        validate(args, epoch)
        print('validate done')
        save_logic(args, epoch)
        args.epochal_update(args, epoch)


