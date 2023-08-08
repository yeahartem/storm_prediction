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
            nn.Linear(400, 256),
            nn.ReLU(),  
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
        
    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output
    


class WindNet41x41(nn.Module):
    def __init__(self) -> None:        
        super(WindNet41x41, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=64, kernel_size=(5, 5, 5)), 
            nn.ReLU(),
            nn.InstanceNorm3d(64),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.InstanceNorm3d(64),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3)),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Flatten(start_dim=1),
            nn.Linear(2304, 256),
            nn.ReLU(),  
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
        
    def forward(self, X) -> torch.Tensor:
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
        output = self.net(X)
        return output


class WindNetElev41x41(nn.Module):
     def __init__(self) -> None:        
         super(WindNetElev41x41, self).__init__()
         self.net_climate = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=90, kernel_size=(5, 5, 5)), 
            nn.ReLU(),
            nn.InstanceNorm3d(90),
            nn.Conv3d(in_channels=90, out_channels=90, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.InstanceNorm3d(90),
            nn.Conv3d(in_channels=90, out_channels=45, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(45),
            nn.Conv3d(in_channels=45, out_channels=45, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(45),
            nn.Conv3d(in_channels=45, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3)),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            nn.Flatten(start_dim=1),
            nn.Linear(2304, 256)
            )
         self.net_elevation1 = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(51, 51), dilation=10),
             nn.ReLU(),
             nn.BatchNorm2d(32),
             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(10, 10), dilation=1, stride=3),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(26, 26), dilation=5, stride=10),
             nn.Flatten(),
             nn.Linear(3969, 256),
             )
         net_elevation2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), dilation=1),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), dilation=1),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=1),
            nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(26, 26), dilation=5, stride=10),
            nn.Flatten(start_dim=1),
            nn.Linear(3969, 256),
            )
         self.net_combined = nn.Sequential(
             nn.ReLU(),  
             nn.BatchNorm1d(256),
             nn.Linear(256, 1)
         )
     def forward(self, X) -> torch.Tensor:
         clim = self.net_climate(X[0])
         elev = self.net_elevation2(X[1])
         output = self.net_combined(clim + elev)
         return output
     

