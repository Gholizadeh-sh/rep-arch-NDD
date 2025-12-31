# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogReg(nn.Module):

    def __init__(self, num_rows: int, block_size: int = 64):
        super().__init__()
        input_dim = num_rows * block_size
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, num_rows, 64)
        batch_size = x.size(0)
        out = x.view(batch_size, -1)  # (batch, input_dim)
        logit = self.linear(out)  # (batch, 1)
        return logit.squeeze(-1)  # (batch,)


class MLP(nn.Module):

    def __init__(self, num_rows: int, block_size: int = 64):
        super().__init__()
        input_dim = num_rows * block_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (batch, num_rows, 64)
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        logit = self.net(out)
        return logit.squeeze(-1)


class Conv1D(nn.Module):

    def __init__(self, num_rows: int, block_size: int = 64, base_channels: int = 64):
        super().__init__()
        self.conv_in = nn.Conv1d(num_rows, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.fc = nn.Linear(base_channels, 1)

    def forward(self, x):
        # x: (batch, num_rows, 64)
        out = self.conv_in(x)  # (batch, C, 64)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = out.mean(dim=2)  # global average pooling over length
        logit = self.fc(out)
        return logit.squeeze(-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):

    def __init__(
        self,
        num_rows: int,
        block_size: int = 64,
        base_channels: int = 64,
        num_blocks: int = 5,
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(num_rows, base_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(base_channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock1D(base_channels, kernel_size=3))
        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Linear(base_channels, 1)

    def forward(self, x):
        # x: (batch, num_rows, 64)
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)

        out = self.blocks(out)

        out = out.mean(dim=2)  # GAP
        logit = self.fc(out)
        return logit.squeeze(-1)
