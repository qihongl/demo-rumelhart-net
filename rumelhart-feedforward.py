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
    [1,1,0,1,0,0,0, 0,0,0,0,0,0,0],  # Canary
    [1,1,0,0,1,0,0, 0,0,0,0,0,0,0],  # Robin
    [1,0,1,0,0,1,0, 0,0,0,0,0,0,0],  # Shark
    [1,0,1,0,0,0,1, 0,0,0,0,0,0,0],  # Salmon
    [0,0,0,0,0,0,0, 1,1,0,1,0,0,0],  # Oak
    [0,0,0,0,0,0,0, 1,1,0,0,1,0,0],  # Pine
    [0,0,0,0,0,0,0, 1,0,1,0,0,1,0],  # Rose
    [0,0,0,0,0,0,0, 1,0,1,0,0,0,1],  # Daisy
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

n_epochs = 2000
hidden_size = 64
n_features = np.shape(targets)[-1]

# 3. Train the model
model = Net(hidden_size=hidden_size, output_dim=n_features)
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
# ax.set_title('Inputs')
ax.set_xticks(np.arange(-.5, n_features, 1), minor=True)
ax.set_yticks(np.arange(-.5, n_items, 1), minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
f.tight_layout()


# loss
f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
f.tight_layout()

# show the input-input similarity matrix
f, ax = plt.subplots(1, 1, figsize=(8, 4))
# sns.heatmap(np.cov(targets), ax=ax, cmap='coolwarm',square=True)
sns.heatmap(np.corrcoef(targets), ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
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
# sns.heatmap(np.cov(all_hidden[-1]), ax=ax, cmap='coolwarm', square=True)
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
mds = MDS(n_components=3)
transformed = mds.fit_transform(all_hidden_k_reshaped)
transformed_reshaped = transformed.reshape(len(all_hidden_k), n_items, 3)

# plot the trajectories
f, ax = plt.subplots(1, 1, figsize=(8, 6))  # Made figure wider to accommodate legend
colors = sns.color_palette("colorblind", n_items)
for i in range(n_items):
    x = transformed_reshaped[:, i, 0]
    y = transformed_reshaped[:, i, 1]
    ax.plot(x, y, label=items[i], color=colors[i])
    ax.scatter(x[0], y[0], marker='o', color='red')
ax.legend()
ax.set_xlabel('MDS dim 1')
ax.set_ylabel('MDS dim 2')
ax.set_title('Hidden Representation During Training')
plt.tight_layout()

# # Create and save legend separately
# figlegend = plt.figure(figsize=(8, 1.2))
# # Get the legend handles and labels from the main plot
# handles, labels = ax.get_legend_handles_labels()
# # Create the legend with 2 rows
# legend = figlegend.legend(handles, labels, loc='center', ncol=4, mode="expand")
# figlegend.savefig('mds_legend.png', bbox_inches='tight', dpi=150)
#
# # make a video of the MDS results in 3d and rotate it 360 degrees
# from matplotlib.animation import FuncAnimation
#
# # Create a 3D plot
# fig = plt.figure(figsize=(8, 6))  # Adjusted size since no legend
# ax = fig.add_subplot(111, projection='3d')
#
# scatter = None
# lines = []
# # Using the same colors as defined above
#
# def init():
#     global scatter, lines
#     # Plot scatter points for current position
#     scatter = ax.scatter(transformed_reshaped[0, :, 0],
#                         transformed_reshaped[0, :, 1],
#                         transformed_reshaped[0, :, 2],
#                         c=colors)
#
#     # Plot trajectories for each item
#     lines = []
#     for i in range(n_items):
#         line, = ax.plot(transformed_reshaped[:, i, 0],
#                        transformed_reshaped[:, i, 1],
#                        transformed_reshaped[:, i, 2],
#                        color=colors[i],
#                        alpha=0.5)  # Removed label parameter
#         lines.append(line)
#
#     ax.set_xlabel('MDS dim 1')
#     ax.set_ylabel('MDS dim 2')
#     ax.set_zlabel('MDS dim 3')
#     return [scatter] + lines
#
# def update(frame):
#     ax.view_init(elev=30, azim=frame)
#     return [scatter] + lines
#
# # Create the animation
# animation = FuncAnimation(fig, update, frames=range(0, 360, 1),
#                         init_func=init, blit=True, interval=5)  # 1000ms/50fps = 20ms per frame
#
# # Save the animation as a gif
# animation.save('mds_animation.gif', writer='pillow', fps=120)
#
# plt.show()
