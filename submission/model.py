from torch import nn, optim
from math import sqrt

class Model(torch.nn.Module):
    # TODO: Define your model here!

    def __init__(self):
        """Constructor for the neural network."""
        super(Model, self).__init__()        # Call superclass constructor

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(in_features=512, out_features=2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        return x
