from starter_code.dataset import get_loaders
from starter_code.train import train_model, plot_performance, load_model
from dataset import preprocess
from model import Model
train_loader, val_loader = get_loaders(preprocessor=preprocess)
import torch

model = Model()

# train the classifier layer first with higher learning rate
for name, param in model.named_parameters():
    if '_fc' not in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=1)
plot_performance(history)

# Unfreeze some feature layers and train on lower learning rate

for name, param in model.named_parameters():
    if '_blocks.15' in name and 'bn' not in name:
        param.requires_grad = True
      
model.source_model._conv_head.weight.requires_grad = True
    
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

weight_decay = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)
plot_performance(history)
