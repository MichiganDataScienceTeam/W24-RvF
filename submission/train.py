from starter_code.dataset import get_loaders
from starter_code.train import train_model, plot_performance, load_model
from dataset import preprocess
from model import Model
train_loader, val_loader = get_loaders(preprocessor=preprocess)
import torch

model = Model()
learning_rate = 0.0001
weight_decay = 0.000001
optimizer = torch.optim.AdamW(params= model.parameters(), lr=learning_rate, weight_decay=weight_decay) # TODO: Change the optimizer to explore different options
criterion = torch.nn.CrossEntropyLoss() # TODO: Change the criterion to explore different options

history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=30)
plot_performance(history)

# Load the model from the training run
load_model(model, "checkpoints", 0) # change epoch from 0 to something else
