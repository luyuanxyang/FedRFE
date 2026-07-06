

import torch
import torch.nn as nn
import math

def compute(padding,kernel_size,stride):

    conv_size = math.floor(
        (32 + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    )
    pool_out_size = math.floor((conv_size - 4) / 4 + 1)
    flat_dim = 64 * pool_out_size * pool_out_size

    return flat_dim

class RandomFeaturesCNN(nn.Module):

    def __init__(self, feat_dim=256, kernel_size=3,stride=1):
        super().__init__()

        # --- random (frozen) feature extractor ---
        # self.conv1 = nn.Conv2d(3,64, 3, padding=1)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2
        )
        self.size=compute(kernel_size//2,kernel_size,stride)

        self.fc_feat = nn.Linear(self.size, feat_dim)

        self.pool = nn.MaxPool2d(4)                   # /2 spatial
        self.relu = nn.ReLU(inplace=True)

        # Freeze extractor params
        for p in [*self.conv1.parameters(),
                  *self.fc_feat.parameters()]:
            p.requires_grad = False

    def forward(self, x):
        # Feature extractor (frozen)
        with torch.no_grad():
            x = self.relu(self.conv1(x))
            x = self.pool(x)   
            
            x = x.view(x.size(0), -1)     # flatten
            x = self.relu(self.fc_feat(x))# (N, feat_dim)

        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim=2560, out_dim=3000,num_classes=100, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)     # feature

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=False)
        self.head=nn.Linear(out_dim,num_classes)

    def forward(self, x):
        x = self.act(self.fc1(x))
        h=self.drop(x)

        return self.head(h)