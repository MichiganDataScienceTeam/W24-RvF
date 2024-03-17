from torch import nn, optim
class Model(torch.nn.Module):
    # TODO: Define your model here!
    def __init__(self):
        """Constructor for the neural network."""
        super(Model, self).__init__()        # Call superclass constructor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()              
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(373248, 10) 

    def forward(self, x):
        z1 = self.conv1(x)
        h1 = self.relu(z1)
        p1 = self.pool(h1)

        z2 = self.conv2(p1)
        h2 = self.relu(z2)
        p2 = self.pool(h2)

        flat = self.flatten(p2)
        z = self.fc(flat)

        return z
