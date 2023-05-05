from torch import nn
import torch
import wandb

class WindNet(nn.Module):
    def __init__(self, cfg) -> None:        
        super(WindNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels=len(cfg.variables), out_channels=32, kernel_size=(7, 5, 5)), 
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Flatten(start_dim=1),
            nn.Linear(1024, 100),
            nn.ReLU(),  
            nn.BatchNorm1d(100),
            nn.Linear(100, 1)
        )

    def forward(self, X) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.net(X)
        return output
