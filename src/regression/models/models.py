from torch import nn
import torch
import logging
import timm

class WindNet27x47(nn.Module):
    def __init__(self) -> None:        
        super(WindNet27x47, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=120, kernel_size=(1, 5, 5)), 
            nn.GELU(),
            nn.InstanceNorm3d(120),
            nn.Conv3d(in_channels=120, out_channels=120, kernel_size=(1, 5, 5), groups=120),
            nn.Conv3d(in_channels=120, out_channels=120, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(120),
            nn.Conv3d(in_channels=120, out_channels=120, kernel_size=(1, 5, 5), groups=120),
            nn.Conv3d(in_channels=120, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.MaxPool3d((1, 3, 3), stride=(2, 2, 2)),
            nn.InstanceNorm3d(180),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 3, 3), groups=180),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 3, 3), groups=180),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)))
        
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=60, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block5 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block6 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block7 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        self.block8 = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=180, padding=1),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(3, 3, 3), groups=90, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=180, out_channels=180, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(180))
        
        self.head = nn.Sequential(
            nn.Conv3d(in_channels=180, out_channels=90, kernel_size=(3, 3, 3), groups=90),
            nn.GELU(),
            nn.InstanceNorm3d(90),
            nn.Conv3d(in_channels=90, out_channels=90, kernel_size=(3, 3, 3), groups=10),
            nn.MaxPool3d((2, 3, 3), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=90, out_channels=60, kernel_size=(1, 1, 1)),
            nn.MaxPool3d((1, 3, 3), stride=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(60),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.4),
            nn.Linear(6000, 7),
        )
        
    def forward(self, X) -> torch.Tensor:
        X = self.net1(X)
        X = self.block1(X) + X
        X = self.block2(X) + X
        X = self.block3(X) + X
        X = self.block4(X) + X
        X = self.block5(X) + X
        X = self.block6(X) + X
        X = self.block7(X) + X
        X = self.block8(X) + X
        X = self.head(X)
        return X
    
class GhostWindNet27(nn.Module):
    def __init__(self) -> None:        
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(27*(self.embed + 4), 70),
            nn.ReLU(),
            nn.BatchNorm1d(70),
            nn.Linear(70, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)
        pos = torch.reshape(pos, [b * days, 4])
        X = torch.cat((X, pos), 1)
        X = torch.reshape(X, [b, days * (self.embed + 4)])
        X = self.head(X)
        return X

