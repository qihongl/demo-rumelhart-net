import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='white', palette='colorblind', context='talk')

# 1. Create stimuli
items = ['Sparrow', 'Robin', 'Eagle', 'Hawk', 'Finch', 'Pigeon', 'Banda Myzomela!', 'Penguin']
n_items = len(items)

# Target feature vectors (hierarchical structure)
inputs = torch.tensor([
    [1,1,1,0,0,0,0,0,0,0],
    [1,1,0,1,0,0,0,0,0,0],
    [1,1,0,0,1,0,0,0,0,0],
    [1,1,0,0,0,1,0,0,0,0],
    [1,1,0,0,0,0,1,0,0,0],
    [1,1,0,0,0,0,0,1,0,0],
    [1,1,0,0,0,0,0,0,1,0],
    [1,1,0,0,0,0,0,0,0,1],
], dtype=torch.float32)

# Target features: [can_fly, is_big]
targets = torch.tensor([
    [1],  # Sparrow
    [1],  # Robin
    [1],  # Eagle
    [1],  # Hawk
    [1],  # Finch
    [1],  # Pigeon
    [1],  # Banda Myzomela
    [0],  # Penguin
], dtype=torch.float32)

# show the inputs
f, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.imshow(inputs.numpy())
ax.set_yticks(range(n_items))
ax.set_yticklabels(items)
ax.set_xlabel('Features')
ax.set_ylabel('Items')
f.tight_layout()


# 2. Define MLP with one hidden layer
class Net(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        self.initialize_weights()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

    def initialize_weights(self, scale=0.01):
        for param in self.parameters():
            param.data.uniform_(-scale, scale)

# Initialize model and training parameters
np.random.seed(0)
torch.manual_seed(0)
hidden_size = 16
output_dim = 2
n_features = np.shape(inputs)[-1]
model = Net(hidden_size, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training indices
train_indices_phase1 = [0, 1, 2, 3, 4, 5]  # 6 regular birds
train_indices_phase2 = [7]                  # Penguin
train_indices_phase3 = [0, 1, 2, 3, 4, 5, 7]  # all except for bird 6

n_epochs_phase1 = n_epochs_phase2 = n_epochs_phase3 = 500

# Phase 1: Train on regular birds
losses_phase1 = []
for epoch in range(n_epochs_phase1):
    outputs = model(inputs[train_indices_phase1])
    loss = criterion(outputs, targets[train_indices_phase1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_phase1.append(loss.item())

# Evaluate after Phase 1
with torch.no_grad():
    predictions_phase1 = model(inputs)
    print("After Phase 1 (Training on 6 regular birds):")
    for i, item in enumerate(items):
        fly_prob = predictions_phase1[i, 0].item()
        print(f"{item}: Can Fly = {fly_prob:.2f}")

# Phase 2: Train on penguin only
losses_phase2 = []
for epoch in range(n_epochs_phase2):
    outputs = model(inputs[train_indices_phase2])
    loss = criterion(outputs, targets[train_indices_phase2])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_phase2.append(loss.item())

# Evaluate after Phase 2
with torch.no_grad():
    predictions_phase2 = model(inputs)
    print("\nAfter Phase 2 (Training on Penguin):")
    for i, item in enumerate(items):
        fly_prob = predictions_phase2[i, 0].item()
        print(f"{item}: Can Fly = {fly_prob:.2f}")

# Plot training losses
for data, label in zip([losses_phase1, losses_phase2], ['Phase 1 - Regular birds', 'Phase 2 - Penguin']):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(data)
    ax.set_title(f'{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    f.tight_layout()

width = .7
for data, title in zip([predictions_phase1, predictions_phase2], ['Phase 1', 'Phase 2']):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.barplot(y=items, x=data[:,0].numpy(), width=width, ax=ax, orient='horizontal')
    ax.set_xlabel('"Fly Probability"')
    ax.set_title(f'After {title.lower()} training')
    ax.set_xlim(0, 1.05)
    # ax.legend()
    f.tight_layout()
    sns.despine()


# phase 3: train on all items in an interleaved fashion
losses_phase3 = []
for epoch in range(n_epochs_phase3):
    outputs = model(inputs[train_indices_phase3])
    loss = criterion(outputs, targets[train_indices_phase3])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_phase3.append(loss.item())

# Evaluate after Phase 3
with torch.no_grad():
    predictions_phase3 = model(inputs)
    print("\nAfter Phase 3 (Training on all items):")
    for i, item in enumerate(items):
        fly_prob = predictions_phase3[i, 0].item()


# Plot training losses
for data, label in zip([losses_phase1], ['Phase 3']):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(data)
    ax.set_title(f'{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    f.tight_layout()
    sns.despine()

# Plot predictions
for data, title in zip([predictions_phase3], ['Phase 3']):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.barplot(y=items, x=data[:,0].numpy(), width=width, ax=ax, orient='horizontal')
    ax.set_xlabel('"Fly Probability"')
    ax.set_title(f'After {title.lower()} training')
    ax.set_xlim(0, 1.05)
    f.tight_layout()
    sns.despine()
