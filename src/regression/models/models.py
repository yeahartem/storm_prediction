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
    

class Linear83x5(nn.Module):
    def __init__(self) -> None:   
        super(Linear83x5, self).__init__()        

        self.net = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=3, kernel_size=(3, 3, 3), dilation=2, stride=1), 
            nn.Flatten(start_dim=1), 
            nn.Linear(5925, 1),
        )

    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output


class WindNetElev83x41(nn.Module):
     def __init__(self) -> None:        
         super(WindNetElev83x41, self).__init__()
         self.net_climate_p1 = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=140, kernel_size=(5, 5, 5), dilation=(2, 2, 2), stride=2), 
            nn.ReLU(),
            nn.InstanceNorm3d(140),
            nn.Conv3d(in_channels=140, out_channels=120, kernel_size=(5, 5, 5), dilation=(2, 2, 2), stride=1),
            nn.ReLU(),
            nn.InstanceNorm3d(120),
            nn.Conv3d(in_channels=120, out_channels=90, kernel_size=(5, 5, 5), dilation=(2, 2, 2), stride=1),
            nn.ReLU(),
            nn.InstanceNorm3d(90),
            nn.Conv3d(in_channels=90, out_channels=90, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1)),
            nn.InstanceNorm3d(90),
            nn.Conv3d(in_channels=90, out_channels=45, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(45),
            )
         self.net_climate_p2  = nn.Sequential(
            nn.Conv3d(in_channels=45, out_channels=45, kernel_size=(3, 3, 3), dilation=2, stride=1),
            nn.ReLU(),
            nn.InstanceNorm3d(45),
            nn.Conv3d(in_channels=45, out_channels=45, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(45),
            nn.Conv3d(in_channels=45, out_channels=25, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(25),
            nn.Conv3d(in_channels=25, out_channels=25, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.InstanceNorm3d(25),
            nn.Flatten(start_dim=1),
            nn.Linear(1600, 256))

         self.net_elevation2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=90, kernel_size=(12, 12), dilation=3, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(90),
            nn.Conv2d(in_channels=90, out_channels=64, kernel_size=(10, 10), dilation=3, stride=2),
            nn.MaxPool2d((3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(10, 10), dilation=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), dilation=2),
            nn.MaxPool2d((3, 3), stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), dilation=2),
            nn.MaxPool2d((3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), dilation=2),
            nn.MaxPool2d((3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Flatten(start_dim=1),            
            nn.Linear(64, 64),
            )
         self.net_combined = nn.Sequential(
             nn.BatchNorm1d(320),
             nn.Dropout(p=0.2),
             nn.Linear(320, 1)
         )
     def forward(self, X) -> torch.Tensor:
         clim = self.net_climate_p1(X[0])
         clim = self.net_climate_p2(clim)
         elev = self.net_elevation2(X[1])
         #comb = torch.empty(320, dtype=torch.float16)

         output = self.net_combined(torch.cat((clim,elev), dim=1))
         return output
