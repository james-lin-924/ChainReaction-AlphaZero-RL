import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, grid_size=5, hidden_dim=256):
        super(AlphaZeroNet, self).__init__()
        
        self.input_dim = grid_size * grid_size * 2  # 5x5 grid, 2 layers (Atoms, Owners)
        self.action_size = grid_size * grid_size
        
        # --- Shared Hidden Layers (Feature Extractor) ---
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # --- Policy Head ---
        # Outputs a probability distribution over all moves
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        
        # --- Value Head ---
        # Outputs a scalar value [-1, 1] evaluation of the board
        self.value_head_fc = nn.Linear(hidden_dim, 64)
        self.value_head_out = nn.Linear(64, 1)

    def forward(self, x):
        # x input shape: (Batch, 2, 5, 5) or (Batch, 5, 5, 2) depending on preprocessing
        # We need to flatten it to (Batch, 50)
        
        x = x.reshape(x.size(0), -1)  # Flatten
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        # Policy Head
        p = self.policy_head(x)
        p = F.log_softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_head_fc(x))
        v = torch.tanh(self.value_head_out(v))
        
        return p, v