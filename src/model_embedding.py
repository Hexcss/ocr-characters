import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class NeuroOCR(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, use_stn=True):
        super(NeuroOCR, self).__init__()
        self.use_stn = use_stn

        # --- 1. STN ---
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.GELU()
        )

        # For 28x28 input, localization output is [B, 32, 3, 3] -> 288
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # --- 2. Backbone ---
        self.conv_in = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(32, 64, stride=2)   # 14x14
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 7x7
        self.layer3 = ResidualBlock(128, 256, stride=1) # 7x7

        self.flatten_dim = 256 * 7 * 7
        self.fc_features = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.4)
        self.embedding_head = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def stn(self, x):
        xs = self.localization(x)
        # Use flatten instead of view to be robust to channels_last / non-contiguous
        xs = torch.flatten(xs, 1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)

        x = F.gelu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten robustly (channels_last safe)
        x = torch.flatten(x, 1)
        x = F.gelu(self.fc_features(x))
        x = self.dropout(x)

        embedding = self.embedding_head(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        output = self.classifier(embedding)
        return output, embedding
