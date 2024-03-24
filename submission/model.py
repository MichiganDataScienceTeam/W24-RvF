import torch

class Model(torch.nn.Module):
    def __init__(self):
      """Constructor for the neural network."""
      super(Model, self).__init__()        # Call superclass constructor
      self.batchnorm = torch.nn.BatchNorm2d(num_features = 4)
      self.padding = torch.nn.ZeroPad2d(padding = 2)
      self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, stride = 1)
      self.dropout = torch.nn.Dropout(p = 0.05)
      self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 128, kernel_size = 3, stride = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
      self.relu = torch.nn.ReLU()
      self.flatten = torch.nn.Flatten()
      self.fc = torch.nn.Linear(540800, 10)

    def forward(self, x):
      b1 = self.batchnorm(x)
      pa1 = self.padding(b1)

      z1 = self.conv1(pa1)
      d1 = self.dropout(z1)
      h1 = self.relu(d1)
      p1 = self.pool(h1)

      pa2 = self.padding(p1)

      z2 = self.conv2(pa2)
      d2 = self.dropout(z2)
      h2 = self.relu(d2)
      p2 = self.pool(h2)

      flat = self.flatten(p2)
      z = self.fc(flat)

      return z
