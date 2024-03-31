
train_loader, val_loader = get_loaders(preprocessor=preprocess)
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

# Unfreeze some feature layers

for name, param in model.named_parameters():
    if '_blocks.22' in name and 'bn' not in name:
        param.requires_grad = True

model.source_model._conv_head.weight.requires_grad = True

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

train_loader, val_loader = get_loaders(preprocessor=preprocess2)

weight_decay = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=30)
plot_performance(history)

# Load the model from the training run
load_model(model, "checkpoints", 0) # change epoch from 0 to something else
