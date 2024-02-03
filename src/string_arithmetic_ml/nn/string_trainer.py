import torch.nn as nn
import string_arithmetic_ml.nn.string_dataset as data
import string_arithmetic_ml.prep.string_arithmetic_generator as generator
from torch.utils.data import DataLoader

# Define dataset and data loader
training_set, testing_set, validation_set = data.split_dataset(data.StringDataset(generator.default_save_path), .75, .15)

training_loader = DataLoader(testing_set, batch_size=2**5, shuffle=True, drop_last=True)
testing_loader = DataLoader(testing_set, batch_size=2**5, shuffle=True, drop_last=True)
validation_loader = DataLoader(testing_set, batch_size=2**5, shuffle=True)

# Create the model
# model = arch.StringRNN()

# Define a loss function
error = nn.MSELoss()

# Define optimizer
learning_rate = 0.05
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Arbitrary Stop Point For Seeing if this works
# num_epochs = 5
# if __name__ == '__main__':
#     for epoch in range(num_epochs):
#         for equation, solution in training_loader:
#             optimizer.zero_grad()
#             predicted_output = model(equation)
#             loss_value = error(predicted_output, solution)
#             # compute the gradient
#             loss_value.backward()
#             # update the model
#             optimizer.step()
#
#     print(x := gs.generate_dataset(1))
#     print(model(x[0]))
#     print(x[1])
