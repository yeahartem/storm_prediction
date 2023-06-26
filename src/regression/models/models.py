from torch import nn
import torch


class WindNet20x41(nn.Module):
    def __init__(self) -> None:        
        super(WindNet20x41, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=64, kernel_size=(7, 5, 5)), 
            nn.ReLU(),
            nn.InstanceNorm3d(64),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.MaxPool3d((5, 3, 3), stride=(4, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 100),
            nn.ReLU(),  
            nn.BatchNorm1d(100),
            nn.Linear(100, 1)
        )
        
    def forward(self, X) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.net(X)
        return output
    

class Linear10x51(nn.Module):
    def __init__(self) -> None:   
        super(Linear10x51, self).__init__()        

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(134946, 200),
            nn.ReLU(),  
            nn.BatchNorm1d(200),
            nn.Linear(200, 1),
        )

    def forward(self, X) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.net(X)
        return output