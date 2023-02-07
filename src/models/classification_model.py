from torch import nn
import torch


class WindNet(nn.Module):
    def __init__(self, args) -> None:
        super(WindNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.args = args
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.BatchNorm2d(128),
            self.conv2,
            nn.ReLU(),
            nn.BatchNorm2d(64),
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.BatchNorm1d(9 * 32),
            nn.Linear(9 * 32, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, X) -> torch.Tensor:
        output = self.net(X.float())
        return output
