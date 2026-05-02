from torch import nn
import torch
import logging
import timm
    
# class GhostWindNet27(nn.Module):
#     def __init__(self) -> None:        
#         super(GhostWindNet27, self).__init__()
#         self.embed = 70
#         self.time_window = 27
#         self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False, in_chans=4)
#         self.head1 = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(self.time_window*(self.embed + 4), 70),
#             # nn.ReLU(),
#             nn.LeakyReLU(inplace=True),
#             nn.BatchNorm1d(70),
#             nn.Linear(70, 1),
#             # nn.Linear(70, 7), # для Quantile Regression квантильная регрессия
#         )

#     def forward(self, X) -> torch.Tensor:
#         X, pos = X
#         b = X.shape[0]
#         days = X.shape[2]
#         X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
#         X = self.ghostnetv2(X)
#         pos = torch.reshape(pos, [b * days, 4])
#         X = torch.cat((X, pos), 1)
#         X = torch.reshape(X, [b, days * (self.embed + 4)])
#         X = self.head1(X)
#         return X

class GhostWindNet27(nn.Module):
    """Wind/storm classifier on a 27-day stack of (in_chans, 95, 95) CMIP6 patches.

    NOTE (2026-05-01): pos = [time_pos, time_pos_m, lat/90, lon/180] is NO LONGER
    concatenated with the backbone embedding. Previously those 4 numbers were
    appended to head_lin1 and gave the model direct access to (station, month)
    climatology – so all ablations collapsed to ~AUROC 0.845 regardless of the
    CMIP6 channels. The model now has to learn the physical signal from the
    backbone alone. pos is still threaded through forward() because test_step
    in pl_module.py reads it for regional/seasonal aggregation.
    """

    def __init__(self, in_chans=4, use_pos_in_head: bool = False, drop_path_rate: float = 0.0) -> None:
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.time_window = 27
        self.use_pos_in_head = bool(use_pos_in_head)
        # Stochastic depth (drop_path) regularises deep backbones with virtually
        # no compute overhead. Older versions of timm's GhostNetV2 don't accept
        # this kwarg — gracefully fall back to default.
        backbone_kwargs = dict(
            num_classes=self.embed,
            pretrained=False,
            in_chans=in_chans,
        )
        if float(drop_path_rate) > 0.0:
            try:
                self.ghostnetv2 = timm.create_model(
                    'ghostnetv2_160', drop_path_rate=float(drop_path_rate), **backbone_kwargs
                )
            except TypeError as e:
                logging.warning(
                    f"timm.create_model('ghostnetv2_160', drop_path_rate=...) failed: {e}. "
                    f"Falling back to default (no stochastic depth). Upgrade timm to enable it."
                )
                self.ghostnetv2 = timm.create_model('ghostnetv2_160', **backbone_kwargs)
        else:
            self.ghostnetv2 = timm.create_model('ghostnetv2_160', **backbone_kwargs)

        # --- Мы разбираем head1 на отдельные слои для отладки ---
        self.head_dropout = nn.Dropout(0.4)
        head_in = self.time_window * (self.embed + 4) if self.use_pos_in_head else self.time_window * self.embed
        self.head_lin1 = nn.Linear(head_in, 70)

        # Используем LeakyReLU, как и обсуждали, чтобы сразу проверить гипотезу
        self.head_activation = nn.LeakyReLU(inplace=True)
        # self.head_bn = nn.BatchNorm1d(70)
        self.head_lin2 = nn.Linear(70, 1)

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)                              # (b*days, embed)

        if self.use_pos_in_head:
            # Legacy path — kept ONLY for reproducing pre-fix behaviour. Concatenating
            # 27 copies of [lat, lon, time_pos] gives the head a direct climatology
            # lookup, so backbone information becomes irrelevant. Default is False.
            pos = torch.reshape(pos, [b * days, 4])
            X = torch.cat((X, pos), 1)                      # (b*days, embed+4)
            X = torch.reshape(X, [b, days * (self.embed + 4)])
        else:
            X = torch.reshape(X, [b, days * self.embed])    # (b, days*embed)

        X = self.head_dropout(X)
        X = self.head_lin1(X)
        X = self.head_activation(X)
        X = self.head_lin2(X)

        return X

# Эта строчка нужна, чтобы модель имела доступ к self.trainer.global_step
# Добавь её в класс WindNetPL в файле pl_module.py в метод __init__
# self.net.trainer = self.trainer

class BaselineQW(nn.Module):
    def __init__(self) -> None:        
        super(BaselineQW, self).__init__()
        self.dummy = nn.Linear(1, 1)
    def forward(self, X) -> torch.Tensor:
        X, pos = X
        
        X = X[:, 0, :, 1, 1]
        X = (X * 4.7078495) +  8.934635 + self.dummy(torch.tensor([1.1])) * 0.0
        Q = torch.stack([torch.quantile(X, q=0.96, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.85, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.70, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.50, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.25, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.15, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.05, interpolation='linear', axis=1)], axis=1)      
        
        return Q
    
class BaselineQT(nn.Module):
    def __init__(self) -> None:        
        super(BaselineQT, self).__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        
        X = X[:, 1, :, 1, 1]
        X = (X * 20.841803) +  280.37646 - 271.15 + self.dummy(torch.tensor([1.1])) * 0.0
        Q = torch.stack([torch.quantile(X, q=0.96, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.85, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.70, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.50, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.25, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.15, interpolation='linear', axis=1),
                          torch.quantile(X, q=0.05, interpolation='linear', axis=1)], axis=1)
        return Q


class BaselineLinear(nn.Module):
    def __init__(self) -> None:
        super(BaselineLinear, self).__init__()
        self.Lin = nn.Sequential(
            nn.Linear(27, 27),
            nn.BatchNorm1d(27),
            nn.Linear(27, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X

        X = X[:, 0, :, 1, 1]
        X = self.Lin(X)
        return X
