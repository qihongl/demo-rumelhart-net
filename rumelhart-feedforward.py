import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import seaborn as sns

sns.set(style='white', palette='colorblind', context='talk')

# 1. Create stimuli
items = ['Canary', 'Robin', 'Shark', 'Salmon', 'Oak', 'Pine', 'Rose', 'Daisy']
n_items = len(items)
# Input (one-hot vectors)

inputs = torch.eye(n_items)  # 8x8 tensor

# Target feature vectors (hierarchical structure)
targets = torch.tensor([
    # Animal features            Plant features
    [1,1,0,1, 0,0,0,0],  # Canary
    [1,1,0,0, 0,0,0,0],  # Robin
    [1,0,1,1, 0,0,0,0],  # Shark
    [1,0,1,0, 0,0,0,0],  # Salmon
    [0,0,0,0, 1,1,0,1],  # Oak
    [0,0,0,0, 1,1,0,0],  # Pine
    [0,0,0,0, 1,0,1,1],  # Rose
    [0,0,0,0, 1,0,1,0],  # Daisy
], dtype=torch.float32)



# 2. Define MLP with one hidden layer
class Net(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.hidden = nn.Linear(n_items, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        self.hidden_activations = []
        self.initialize_weights()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        self.hidden_activations.append(x.detach().clone())
        return self.output(x)

    def initialize_weights(self, scale=0.01):
        for param in self.parameters():
            param.data.uniform_(-scale, scale)


'''init the params '''
np.random.seed(0)
torch.manual_seed(0)

n_epochs = 1000
hidden_size = 64
output_dim = np.shape(targets)[-1]
# 3. Train the model
model = Net(hidden_size=hidden_size, output_dim=output_dim)
model.initialize_weights()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Store hidden activations and losses
all_hidden = np.zeros((n_epochs, n_items, hidden_size))
losses = []

for i in range(n_epochs):
    model.hidden_activations = []
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    H = torch.stack(model.hidden_activations).numpy()
    all_hidden[i] = np.squeeze(H)

'''plots'''
# show the inputs
f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.imshow(targets.numpy())
ax.set_yticks(range(n_items))
ax.set_yticklabels(items)
ax.set_xlabel('Features')
ax.set_ylabel('Items')
ax.set_title('Inputs')
f.tight_layout()

# loss
f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
f.tight_layout()

# show the input-input similarity matrix
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(np.corrcoef(targets.T), ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
ax.set_title('Item-item Similarity')
ax.set_xlabel('Items')
ax.set_ylabel('Items')
ax.set_xticks(range(n_items))
ax.set_yticks(range(n_items))
ax.set_xticklabels(items, rotation=90)
ax.set_yticklabels(items, rotation=0)
f.tight_layout()

# show the final hidden-hidden similarity matrix
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(np.corrcoef(all_hidden[-1]), ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
ax.set_title('Representational Similarity')
ax.set_xlabel('Items')
ax.set_ylabel('Items')
ax.set_xticks(range(n_items))
ax.set_yticks(range(n_items))
ax.set_xticklabels(items, rotation=90)
ax.set_yticklabels(items, rotation=0)
f.tight_layout()


# MDS visualization - extract the hidden activations for every other 100 epochs and plot the trajectories
k = 100
all_hidden_k = [all_hidden[i] for i in range(0, len(all_hidden), k)]
all_hidden_k = np.array(all_hidden_k)

all_hidden_k_reshaped = all_hidden_k.reshape(-1, hidden_size)

# Compute MDS
mds = MDS(n_components=2)
transformed = mds.fit_transform(all_hidden_k_reshaped)
transformed_reshaped = transformed.reshape(len(all_hidden_k), n_items, 2)

# plot the trajectories
f, ax = plt.subplots(1, 1, figsize=(8, 7))
colors = plt.cm.tab10(np.linspace(0, 1, n_items))
for i in range(n_items):
    x = transformed_reshaped[:, i, 0]
    y = transformed_reshaped[:, i, 1]
    ax.plot(x, y, label=items[i])
    ax.scatter(x[0], y[0], marker='o', color='red')
    # ax.scatter(x[-1], y[-1], marker='s', color=colors[i])

ax.legend()
ax.set_xlabel('MDS dim 1')
ax.set_ylabel('MDS dim 2')
ax.set_title('Hidden Representation During Training')
f.tight_layout()
