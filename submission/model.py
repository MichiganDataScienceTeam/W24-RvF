import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        source_model = EfficientNet.from_pretrained('efficientnet-b3')

        source_model._fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        self.source_model = source_model

    def forward(self, x):
        x = self.source_model(x)
        return x

