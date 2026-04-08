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
    def __init__(self, in_chans=4) -> None:
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.time_window = 27
        self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False, in_chans=in_chans)
        
        # --- Мы разбираем head1 на отдельные слои для отладки ---
        self.head_dropout = nn.Dropout(0.4)
        self.head_lin1 = nn.Linear(self.time_window * (self.embed + 4), 70)
        # self.head_lin1 = nn.Linear(self.time_window * 4, 70) 

        # Используем LeakyReLU, как и обсуждали, чтобы сразу проверить гипотезу
        self.head_activation = nn.LeakyReLU(inplace=True) 
        # self.head_bn = nn.BatchNorm1d(70)
        self.head_lin2 = nn.Linear(70, 1)

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        # X = torch.reshape(pos, [b, days * 4])
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)
        pos = torch.reshape(pos, [b * days, 4])
        X = torch.cat((X, pos), 1)
        X = torch.reshape(X, [b, days * (self.embed + 4)])

        # ======================= НАЧАЛО БЛОКА ОТЛАДКИ =======================
        # Печатаем только для первых нескольких шагов обучения, чтобы не засорять лог
        # torch.is_grad_enabled() гарантирует, что это происходит только во время обучения
        if torch.is_grad_enabled() and hasattr(self, 'trainer') and self.trainer.global_step % 100 == 0:
            print("\n" + "="*50)
            print(f"--- ДЕБАГ ПРЯМОГО ПРОХОДА (ШАГ {self.trainer.global_step}) ---")
            print(f"Вход в 'голову' | Форма: {X.shape}")
            
            # --- Шаг 1: Dropout ---
            X_drop = self.head_dropout(X)
            
            # --- Шаг 2: Первый Linear слой ---
            X_lin1 = self.head_lin1(X_drop)
            percent_non_positive = (X_lin1 <= 0).float().mean() * 100
            print("\n[ДО АКТИВАЦИИ] Выход из Linear_1:")
            print(f"  Форма: {X_lin1.shape}")
            print(f"  Значения (min, mean, max): {X_lin1.min():.3f}, {X_lin1.mean():.3f}, {X_lin1.max():.3f}")
            print(f"  Процент отрицательных значений: {percent_non_positive:.1f}%")

            # --- Шаг 3: Функция активации (LeakyReLU) ---
            X_act = self.head_activation(X_lin1)
            percent_zeros = (X_act == 0).float().mean() * 100
            print("\n[ПОСЛЕ АКТИВАЦИИ] Выход из LeakyReLU:")
            print(f"  Форма: {X_act.shape}")
            print(f"  Значения (min, mean, max): {X_act.min():.3f}, {X_act.mean():.3f}, {X_act.max():.3f}")
            print(f"  Процент НУЛЕВЫХ значений: {percent_zeros:.1f}%")

            # --- Шаг 4: BatchNorm ---
            # X_bn = self.head_bn(X_act)
            # print("\n[ПОСЛЕ BATCHNORM] Выход из BatchNorm1d:")
            # print(f"  Форма: {X_bn.shape}")
            # print(f"  Значения (min, mean, max): {X_bn.min():.3f}, {X_bn.mean():.3f}, {X_bn.max():.3f}")

            # --- Шаг 5: Второй Linear слой (финальное предсказание) ---
            # X_final = self.head_lin2(X_bn)
            X_final = self.head_lin2(X_act)
            print("\n[ФИНАЛ] Итоговое предсказание:")
            print(f"  Первые 5 предсказаний: {X_final.squeeze()[:5].detach().cpu().numpy().round(3)}")
            print("="*50 + "\n")
            
            # Повторяем вычисления, чтобы вернуть результат
            X = X_final
        else:
            # Обычный проход без print-ов для скорости
            X = self.head_dropout(X)
            X = self.head_lin1(X)
            X = self.head_activation(X)
            # X = self.head_bn(X)
            X = self.head_lin2(X)
        # ======================== КОНЕЦ БЛОКА ОТЛАДКИ ========================
            
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
    
