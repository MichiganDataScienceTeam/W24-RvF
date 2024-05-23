import torch

class Model(torch.nn.Module):
    def __init__(self):
      """Constructor for the neural network."""
      super(Model, self).__init__()        # Call superclass constructor
      self.batchnorm = torch.nn.BatchNorm2d(num_features = 4)
      self.padding = torch.nn.ZeroPad2d(padding = 2)
      self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, stride = 1)
      self.dropout = torch.nn.Dropout(p = 0.10)
      self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 3, stride = 1)
      self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = 3, stride = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
      self.relu = torch.nn.ReLU()
      self.flatten = torch.nn.Flatten()
      self.fc1 = torch.nn.Linear(278784, 10)
      self.fc2 = torch.nn.Linear(10, 1)
      self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
      pa1 = self.padding(x)
      b1 = self.batchnorm(pa1)

      z1 = self.conv1(b1)
      h1 = self.relu(z1)
      p1 = self.pool(h1)

      pa2 = self.padding(p1)

      z2 = self.conv2(pa2)
      h2 = self.relu(z2)
      p2 = self.pool(h2)

      pa3 = self.padding(p2)

      z3 = self.conv3(pa3)
      h3 = self.relu(z3)
      p3 = self.pool(h3)

      flat = self.flatten(p3)
      d1 = self.dropout(flat)
      z1 = self.fc1(d1)
      d2 = self.dropout(z1)
      z2 = self.fc2(d2)

      return self.sigmoid(z2).squeeze(1)
