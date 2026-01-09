import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, grid_size=5, num_res_blocks=4, num_channels=64):
        super(AlphaZeroNet, self).__init__()
        
        self.start_conv = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * grid_size * grid_size, grid_size * grid_size)
        
        # --- Value Head ---
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(grid_size * grid_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x is (Batch, 2, 5, 5)
        
        x = F.relu(self.bn_start(self.start_conv(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        # FIX: Changed .view() to .reshape() to handle non-contiguous tensors safely
        p = p.reshape(p.size(0), -1)  
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v