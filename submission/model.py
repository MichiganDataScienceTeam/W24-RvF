import torch
import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        source_model = EfficientNet.from_pretrained('efficientnet-b7')

        source_model._fc = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 246),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        for param in source_model.parameters():
          param.requires_grad = True
        for param in list(source_model.parameters())[:-20]:
          param.requires_grad = False

        self.source_model = source_model

    def forward(self, x):
        x = self.source_model(x)
        return x
