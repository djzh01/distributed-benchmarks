import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Lambda, Compose
import time

training_data = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=Compose([
        ToTensor(),
        # Lambda(lambda x: x.repeat(3, 1, 1))
    ])
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

learning_rate = 1e-3
batch_size = 64
epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


model = models.GoogLeNet()
model.to(device)
# model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        X = X.to(device)
        y = y.to(device)
        # loss = loss_fn(pred, y)
        loss = loss_fn(pred.logits, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if loss < 0.8:
            return True


start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    if train_loop(train_dataloader, model, loss_fn, optimizer):
        break;
end = time.time()
print(end - start)
