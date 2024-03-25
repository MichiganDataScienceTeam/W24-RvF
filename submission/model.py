import torch
import torchvision.models as models

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.source_model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)

        for name, param in self.source_model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
            else:
                print(f"Unfrozen layer: {name}")
                param.requires_grad = True

        self.source_model.head = torch.nn.Sequential(
            torch.nn.Linear(1280, 2, bias=True)
        )

    def forward(self, x):
        x = self.source_model(x)
        return x
