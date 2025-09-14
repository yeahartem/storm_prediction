import sys,os
sys.path.append(os.getcwd())
from typing import List, Any
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torch.functional import F
import torch.nn as nn
from src.regression.models.models import *
from src.utils.metrics import float_to_binary, float_to_score, get_outliers_s, get_outliers_p
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class WindNetPL(pl.LightningModule):

    def __init__(self, cfg, run_dir=None, eval=False): 
        super().__init__()     
        self.save_hyperparameters()
        self.cfg = cfg        
        self.run_dir = run_dir
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
        if cfg.model_name=="BaselineLinear":
            self.net = BaselineLinear()
        elif cfg.model_name=="BaselineQT":
            self.net = BaselineQT()
        elif cfg.model_name=="BaselineQW":
            self.net = BaselineQW()
        elif cfg.model_name=="GhostWindNet27":
            self.net = GhostWindNet27()
        else:
            raise NotImplementedError(f'Model {cfg.model_name} not found')     
        logging.info(f'Using {cfg.model_name} model')
        if not eval:
            self.scheduler_name = cfg.train.scheduler_name
            if cfg.train.optimizer_name=='AdamW':
                self.optimizer = torch.optim.AdamW
            elif cfg.train.optimizer_name=='RAdam':
                self.optimizer = torch.optim.RAdam
            elif cfg.train.optimizer_name=='SGD':
                self.optimizer = torch.optim.SGD
            else:
                raise NotImplementedError(f'Optimizer {cfg.train.optimizer_name} not found')
            
            if cfg.train.loss_name=='MSELoss':
                self.criterion = torch.nn.MSELoss()
            elif cfg.train.loss_name=='L1Loss':
                self.criterion = torch.nn.L1Loss()
            elif cfg.train.loss_name=='MSELoss_Dense':
                self.criterion = torch.nn.MSELoss(reduction='none')
            elif cfg.train.loss_name=='L1Loss_Dense':
                self.criterion = torch.nn.L1Loss(reduction='none')    
            elif cfg.train.loss_name=='BCELoss':
                print("\n--- ИСПОЛЬЗУЕТСЯ BCELoss С ВЕСАМИ КЛАССОВ ---")
                # Получаем статистику по всему тренировочному датасету
                targets = self.trainer.datamodule.DPL.train_data_idxs[7, :]
                threshold = self.cfg.train.target_threshold
                
                positive_samples = np.sum(targets > threshold)
                negative_samples = len(targets) - positive_samples
                
                # Считаем вес для положительного класса (сильный ветер)
                pos_weight = torch.tensor(negative_samples / positive_samples)
                
                print(f"Статистика для BCELoss:")
                print(f"  Позитивных примеров (> {threshold} м/с): {positive_samples}")
                print(f"  Негативных примеров (<= {threshold} м/с): {negative_samples}")
                print(f"  🔥 Вес для позитивного класса (pos_weight): {pos_weight:.2f}")
                print("--------------------------------------------------\n")
                
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)            
            else:
                raise NotImplementedError(f'Criterion {cfg.train.loss_name} not found')
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_MAE_best = MinMetric()

        self.train_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.val_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.test_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')

        self.test_auroc = torchmetrics.AUROC(task="binary")

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        # self.train_MAE_full = torchmetrics.MeanAbsoluteError() # Quantile Regression
        # self.val_MAE_full = torchmetrics.MeanAbsoluteError()
        # self.test_MAE_full = torchmetrics.MeanAbsoluteError()

        self.val_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.val_recall = torchmetrics.Recall(num_classes=1, task='binary')
        self.test_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.test_recall = torchmetrics.Recall(num_classes=1, task='binary')

        self.train_MAE_OS = torchmetrics.MeanAbsoluteError() # MAE outliers based on station measure
        self.val_MAE_OS = torchmetrics.MeanAbsoluteError() 
        self.test_MAE_OS = torchmetrics.MeanAbsoluteError() 
        
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary')
        self.validation_step_outputs = []


    def forward(self, x):
        return self.net(x)


    def loss(self, y_hat, y, dense_weights):
        # 
        if self.cfg.train.loss_name=='MSELoss_Dense' or self.cfg.train.loss_name=='L1Loss_Dense':
            
            # --- НАЧАЛО ИСПРАВЛЕНИЙ ---

            # Убедимся, что и y_hat, и dense_weights - это 1D векторы
            y_hat_squeezed = y_hat.squeeze()
            dense_weights_squeezed = dense_weights.squeeze()
            
            # 1. Считаем ошибку для каждого примера.
            # self.criterion должен быть инициализирован с reduction='none'
            per_sample_loss = self.criterion(y_hat_squeezed, y)

            # 2. Умножаем ошибку на вес. Теперь оба тензора гарантированно 1D.
            weighted_loss = per_sample_loss * dense_weights_squeezed
            
            # --- КОНЕЦ ИСПРАВЛЕНИЙ ---
            
            # --- ОТЛАДОЧНЫЙ ПРИНТ (оставляем как есть, он полезен) ---
            if self.trainer.global_step % 100 == 0:
                print("\n" + "v"*50)
                print(f"--- ВЗВЕШЕННЫЙ LOSS (ШАГ {self.trainer.global_step}) ---")
                print(f"Истинные значения y (первые 5):   {y[:5].cpu().numpy().round(2)}")
                print(f"Предсказания y_hat (первые 5):   {y_hat_squeezed[:5].cpu().detach().numpy().round(2)}")
                print(f"Веса dense_weights (первые 5):  {dense_weights_squeezed[:5].cpu().detach().numpy().round(2)}")
                print(f"🔥 Взвешенный Loss (первые 5):    {weighted_loss[:5].cpu().detach().numpy().round(2)}")
                print(f"Loss per sample ():     {per_sample_loss}")
                print("^"*50 + "\n")

            # 3. Усредняем взвешенные ошибки.
            return torch.mean(weighted_loss)            
            # # 1. Считаем ошибку для каждого примера отдельно. 
            # #    Результат - тензор такого же размера, как y_hat и y.
            # per_sample_loss = self.criterion(y_hat.squeeze(), y)

            # # 2. Умножаем ошибку каждого примера на его вес.
            # weighted_loss = per_sample_loss * dense_weights
            
            # # --- ОТЛАДОЧНЫЙ ПРИНТ №3 ---
            # if self.trainer.global_step % 50 == 0:
            #     print(f"\n--- DEBUG: loss() step={self.trainer.global_step} ---")
            #     print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}")
            #     print(f"Target y (first 5):      {np.round(y.flatten()[:5].cpu().detach().numpy(), 2)}")
            #     print(f"Prediction y_hat (first 5): {np.round(y_hat.flatten()[:5].cpu().detach().numpy(), 2)}")
                
            #     print(f"Loss per sample (first 5): {np.round(per_sample_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
            #     print(f"Weights (first 5):         {np.round(dense_weights.flatten()[:5].cpu().detach().numpy(), 2)}")
            #     print(f"Weighted loss (first 5): {np.round(weighted_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
            #     print("---------------------------------\n")
            # # --- КОНЕЦ ПРИНТА ---

            # # 3. Теперь усредняем результат, чтобы получить одно число.
            # return torch.mean(weighted_loss)
        elif self.cfg.train.loss_name=='BCELoss':
            # Новая, простая логика. Веса уже "встроены" в self.criterion
            return self.criterion(y_hat.squeeze(), y)
        else:
            # для любой кроме DenseWeight
            per_sample_loss = self.criterion(y_hat.squeeze(), y)
            if self.trainer.global_step % 100 == 0:
                print(f"\n--- DEBUG: loss() step={self.trainer.global_step} ---")
                print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}")
                # y и y_hat здесь должны быть НОРМАЛИЗОВАННЫМИ
                print(f"Target y (norm, first 5):      {y[:5].cpu().numpy().round(2)}")
                print(f"Prediction y_hat (norm, first 5): {y_hat.squeeze()[:5].cpu().detach().numpy().round(2)}")
                print(f"Loss per sample ():     {per_sample_loss}")
                print("---------------------------------\n")                
                
            return per_sample_loss
            # return self.criterion(y_hat, y) # для Quantile Regression квантильная регрессия
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_MAE_best.reset()
        if self.cfg.model_name=="GhostWindNet27":
            self.net.trainer = self.trainer
            
    def model_step(self, batch):
        objs, target, dense_weights = batch
        predictions = self(objs).float()
        # print(objs[0].shape)
        # print(objs[1].shape)
        if self.cfg.train.loss_name == 'BCELoss':
            loss_target = float_to_binary(target, thresh=self.cfg.train.target_threshold).float()
        else:
            loss_target = target.float()        
        loss = self.loss(predictions, loss_target, dense_weights)
        return loss, predictions, target    
    
    def training_step(self, batch, batch_idx):
        # 1. Получаем предсказания модели
        loss, predictions, target = self.model_step(batch)
        
        # <-- ДОБАВЬ ЭТУ ПРОВЕРКУ -->
        if torch.isinf(loss) or torch.isnan(loss):
            logging.warning("!!! Loss is INF or NaN, skipping batch !!!")
        # <-- КОНЕЦ ПРОВЕРКИ -->

        # --- НАЧАЛО БЛОКА ДЛЯ ОТЛОВА СКАЧКОВ MAE ---
        with torch.no_grad(): # Считаем метрику без вычисления градиентов
            batch_mae = torch.nn.functional.l1_loss(predictions.squeeze(), target)
        
        MAE_THRESHOLD = 15.0 # Установи порог, который ты считаешь "аномальным"
        if batch_mae > MAE_THRESHOLD:
            print("\n" + "!"*60)
            print(f"🚨 ОБНАРУЖЕН СКАЧОК MAE НА ШАГЕ {self.trainer.global_step}! MAE = {batch_mae:.2f}")
            print(f"  Истинные значения y: {target.cpu().numpy().round(1)}")
            print(f"  Предсказания y_hat: {predictions.squeeze().cpu().detach().numpy().round(1)}")
            print("!"*60 + "\n")
        # --- КОНЕЦ БЛОКА ---
        
        # 2. Обновляем метрики новыми данными
        self.train_loss(loss)
        
        # =========================== QUANTILE REGRESSION (comment 5 lines below and uncomment those that are lower) =======================================
        preds_squeezed = predictions.squeeze()
        target_squeezed = target
        self.train_MAE(preds_squeezed, target_squeezed) 
        self.train_MAE_OS(*get_outliers_s(preds_squeezed, target_squeezed , thresh=self.cfg.train.target_threshold))
        self.train_AP(float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold),
                    float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold))        
        
        # =========================== QUANTILE REGRESSION  =======================================
        # self.train_MAE(predictions[:, 0], target[:, 0]) # MAE для основного предсказания
        # self.train_MAE_full(predictions, target) # QUANTILE REGRESSION uncomment
        
        # # 3. Самое интересное: превращение регрессии в классификацию
        # # Здесь мы считаем MAE только по тем дням, где реальная скорость ветра была выше порога target_threshold из конфига train_cmip5_test_run.yaml
        # self.train_MAE_OS(*get_outliers_s(predictions[:, 0], target[:, 0] , thresh=self.cfg.train.target_threshold ))
        # # А здесь мы считаем Average Precision
        # self.train_AP(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold ),
        #                 float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))  # target_threshold: 3 - Это значит, что любое значение скорости ветра > 3 м/с считается событием класса "1" (опасный ветер), а всё, что <= 3 — классом "0". Функции float_to_binary и float_to_score в pl_module.py как раз и выполняют это преобразование для подсчета метрик классификации.
        # =========================== QUANTILE REGRESSION =======================================

        # 4. Логируем значения метрик, чтобы их можно было увидеть
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/MAE", self.train_MAE, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/MAE_full", self.train_MAE_full, on_step=True, on_epoch=True, prog_bar=True) # QUANTILE REGRESSION
        self.log("train/MAE_OS", self.train_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/AP", self.train_AP, on_step=True, on_epoch=True, prog_bar=True)
        #if batch_idx%100==0:
        #    self.logger.experiment.log({"train/target": target[:, 0], "train/prediction": predictions[:, 0]})
        #    self.logger.experiment.log({"train/target_50": target[:, 3], "train/prediction_50": predictions[:, 3]})

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output
            

    def validation_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.val_loss(loss)
        
        # =========================== QUANTILE REGRESSION (comment 5 lines below and uncomment those that are lower) =======================================
        preds_squeezed = predictions.squeeze()
        target_squeezed = target
        self.val_MAE(preds_squeezed, target_squeezed)
        self.val_AP(float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold),
                    float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold))    
        self.val_precision(float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold),
                            float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold))
        self.val_recall(float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold),
                            float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold))   
        self.val_MAE_OS(*get_outliers_s(preds_squeezed, target_squeezed , thresh=self.cfg.train.target_threshold))
        
        # =========================== QUANTILE REGRESSION ===========================
        # self.val_MAE(predictions[:, 0], target[:, 0])
        # self.val_MAE_full(predictions, target) # QUANTILE REGRESSION
        # self.val_AP(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold),
        #                     float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))
        # self.val_precision(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold),
        #                     float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))
        # self.val_recall(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold),
        #                     float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))        
        # self.val_MAE_OS(*get_outliers_s(predictions[:, 0], target[:, 0], thresh=self.cfg.train.target_threshold))
        # =========================== QUANTILE REGRESSION ===========================
        self.val_confusion_matrix(float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold),
                            float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold)) 
        
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/MAE", self.val_MAE, on_step=True, on_epoch=True, prog_bar=False)
        # self.log("val/MAE_full", self.val_MAE_full, on_step=True, on_epoch=True, prog_bar=False) # QUANTILE REGRESSION
        self.log("val/MAE_OS", self.val_MAE_OS, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/AP", self.val_AP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

        # if batch_idx%100==0:
        #     self.log("val/prediction_96", predictions[:, 0], on_step=False, on_epoch=True)
        #     self.log("val/prediction_50", predictions[:, 3], on_step=False, on_epoch=True)
        #     self.log("val/prediction_05", predictions[:, 6], on_step=False, on_epoch=True)
            # self.logger.experiment.log({"val/target_96": target[:, 0], "val/prediction_96": predictions[:, 0]})
            # self.logger.experiment.log({"val/target_50": target[:, 3], "val/prediction_50": predictions[:, 3]})
            # self.logger.experiment.log({"val/target_05": target[:, 6], "val/prediction_05": predictions[:, 6]})

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        self.validation_step_outputs.append(output)
        return output
    

    def on_validation_epoch_end(self):
        # Используем наш список, который мы наполнили
        outputs = self.validation_step_outputs
        # Код для расчета val_MAE_best
        MAE = self.val_MAE.compute()
        self.val_MAE_best(MAE)
        self.log("val/MAE_best", self.val_MAE_best.compute(), prog_bar=False)

        # <<< НАЧАЛО БЛОКА, КОТОРЫЙ НУЖНО ДОБАВИТЬ >>>

        # Собираем все предсказания и таргеты из каждого validation_step
        preds = torch.cat([x['preds'] for x in outputs]).cpu()
        targets = torch.cat([x['target'] for x in outputs]).cpu()
        
        print(f"В on_validation_epoch_end() -> preds.shape: {preds.shape}, target.shape: {targets.shape}")
        preds = preds.squeeze() # Comment for Quantile Regression квантильная регрессия

        print("\n--- DEBUG: De-normalization on Epoch End ---")
        print(f"Собранные таргеты (уже реальные): {targets[:5].numpy().round(2)}")
        print(f"Собранные предсказания (): {preds[:5].numpy().round(2)}")
        # print(f"Используемый y_mean: {y_mean:.2f}, y_std: {y_std:.2f}")
        # print(f"Предсказания после де-нормализации (реальные): {preds_real.squeeze()[:5].numpy().round(2)}")
        print("--------------------------------------------\n")
                        
        # Блок для счетчика
        threshold = self.cfg.train.target_threshold
        outlier_count = torch.sum(targets > threshold).item()
        total_samples = targets.numel()
        print(f"\n--- Статистика по выбросам (валидация, скорость > {threshold} м/с) ---")
        print(f"Количество случаев: {outlier_count} из {total_samples} ({outlier_count / total_samples:.2%})")
        
        print("\n--- Анализ результатов по бинам (валидационный набор) ---")
        # Вызываем вашу функцию
        binned_results_df = analyze_performance_by_bins(
            y_pred=preds.numpy().flatten(), 
            y_true=targets.numpy().flatten(),
            n_bins=5
        )
        # Выводим таблицу в консоль в конце каждой эпохи валидации
        print(binned_results_df)
        print("------------------------------------------------------")
        
        # <<< КОНЕЦ БЛОКА, КОТОРЫЙ НУЖНО ДОБАВИТЬ >>>
        
        preds_float = preds
        target_float = targets
        thrs = [0, 3, 5, 8, 10, 12, 15, 17, 20, 23, 25, 27, 30]
        rmses = []
        for th in thrs:
            if len(torch.where(target_float >= th)[0])>0:
                tgt_th = target_float[torch.where(target_float >= th)[0]]
                pred_th = preds_float[torch.where(target_float >= th)[0]]
                rmses.append(np.squeeze(torch.sqrt(torch.mean((pred_th - tgt_th) ** 2)).numpy()))
        
        fig, ax = plt.subplots()
        ax.plot(thrs, rmses, color='purple')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_title(f'RMSE vs Target (Epoch {self.current_epoch})')
        # fig.savefig(os.path.join(self.run_dir, 'RMSE_vs_target.png'))   # save the figure to file  
        # self.logger.experiment.log_figure(fig, f"epoch_{self.current_epoch}_RMSE_vs_target.png")
        # 1. Сначала сохраняем график в файл
        figure_path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}_RMSE_vs_target.png")
        fig.savefig(figure_path)
        # 2. Затем логируем этот файл как артефакт
        self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=figure_path)
        plt.close(fig)

        precision, recall, thresholds = precision_recall_curve(float_to_binary(targets, thresh=self.cfg.train.target_threshold),
                                                            float_to_score(preds, thresh=self.cfg.train.target_threshold)
                    )
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision, color='purple')
        ax_pr.set_title(f'Precision-Recall Curve (Epoch {self.current_epoch})')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_xlabel('Recall')
        # fig.savefig(os.path.join(self.run_dir, 'PR_curve.png'))   # save the figure to file  
        self.logger.experiment.log_figure(run_id=self.logger.run_id,
                                            figure=fig_pr,
                                            artifact_file=f"epoch_{self.current_epoch}_PR_curve.png")  
        plt.close(fig_pr)
        
        # 1. Scatter plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.scatter(target_float.numpy(), preds_float.numpy(), alpha=0.1)
        ax_scatter.plot([target_float.min(), target_float.max()], [target_float.min(), target_float.max()], 'r--', lw=2) # Диагональ y=x
        ax_scatter.set_xlabel('Истинные значения (м/с)')
        ax_scatter.set_ylabel('Предсказанные значения (м/с)')
        ax_scatter.set_title('Предсказание vs. Истина')
        ax_scatter.grid(True)
        # Сохраняем в MLflow
        self.logger.experiment.log_figure(self.logger.run_id, fig_scatter, f"epoch_{self.current_epoch}_scatter_plot.png")
        plt.close(fig_scatter)

        # 2. Гистограмма ошибок
        errors = (preds_float - target_float).numpy()
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(errors, bins=50)
        ax_hist.set_xlabel('Ошибка предсказания (м/с)')
        ax_hist.set_ylabel('Частота')
        ax_hist.set_title('Распределение ошибок')
        self.logger.experiment.log_figure(self.logger.run_id, fig_hist, f"epoch_{self.current_epoch}_error_distribution.png")
        plt.close(fig_hist)

        # 3. Сохранение сырых предсказаний для дальнейшего анализа
        # Это КРАЙНЕ ВАЖНО для воспроизводимости и статистических тестов
        results_df = pd.DataFrame({
            'prediction': preds_float.numpy(),
            'target': target_float.numpy()
        })
        # results_df.to_csv(os.path.join(self.run_dir, f'epoch_{self.current_epoch}_predictions.csv'))
        # self.logger.experiment.log_artifact(os.path.join(self.run_dir, f'epoch_{self.current_epoch}_predictions.csv'))  
        # Путь к файлу лучше сохранить в переменную для читаемости
        csv_path = os.path.join(self.run_dir, f'epoch_{self.current_epoch}_predictions.csv')
        results_df.to_csv(csv_path)
        self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=csv_path)
        
        print("\n--- Матрица ошибок (валидация) ---")
        cm = self.val_confusion_matrix.compute().cpu().numpy()
        print(f"               Предсказано 'Слабый' | Предсказано 'Сильный'")
        print(f"Реально 'Слабый' | {cm[0][0]:<20} | {cm[0][1]:<20} ")
        print(f"Реально 'Сильный'| {cm[1][0]:<20} | {cm[1][1]:<20} ")
        print("------------------------------------")
        # <<< НАЧАЛО БЛОКА ДЛЯ ЛОГИРОВАНИЯ МАТРИЦЫ ОШИБОК >>>
        fig_cm, ax_cm = plt.subplots()
        # Используем imshow для отрисовки матрицы как картинки, cmap='Blues' задает синюю цветовую схему
        im = ax_cm.imshow(cm, cmap='Blues')

        # Добавляем подписи к осям
        ax_cm.set_xticks(np.arange(2))
        ax_cm.set_yticks(np.arange(2))
        ax_cm.set_xticklabels(['Предсказано "Слабый"', 'Предсказано "Сильный"'])
        ax_cm.set_yticklabels(['Реально "Слабый"', 'Реально "Сильный"'])

        # Добавляем цифры в каждую ячейку
        # Этот цикл проходит по каждой ячейке (0,0), (0,1), (1,0), (1,1) и пишет в ней её значение
        for i in range(2):
            for j in range(2):
                # Выбираем цвет текста (белый на тёмном фоне, чёрный на светлом) для читаемости
                text_color = "white" if cm[i, j] > cm.max() / 2. else "black"
                text = ax_cm.text(j, i, cm[i, j],
                            ha="center", va="center", color=text_color)

        ax_cm.set_title(f'Матрица ошибок (Эпоха {self.current_epoch})')
        fig_cm.tight_layout() # Делает график более компактным

        # Логируем и закрываем фигуру, как и с другими графиками
        self.logger.experiment.log_figure(self.logger.run_id, fig_cm, f"epoch_{self.current_epoch}_confusion_matrix.png")
        plt.close(fig_cm)
        tn, fp, fn, tp = cm.flatten() # Распаковываем значения из матрицы

        # Считаем F1-score, избегая деления на ноль
        if (tp + fp == 0) or (tp + fn == 0):
            f1_score = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

        # Логируем F1-score в MLflow
        self.log("val/F1_score", f1_score, on_epoch=True, prog_bar=True)
        # Не забудь сбросить метрику в конце
        self.val_confusion_matrix.reset()

        # <<< ВАЖНО: Очищаем список после использования >>>
        self.validation_step_outputs.clear()

        
    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)
        self.test_loss(loss)
        # =========================== QUANTILE REGRESSION (comment 5 lines below and uncomment those that are lower) =======================================
        preds_squeezed = predictions.squeeze()
        target_squeezed = target
        binary_target = float_to_binary(target_squeezed, thresh=self.cfg.train.target_threshold)
        score_preds = float_to_score(preds_squeezed, thresh=self.cfg.train.target_threshold)
        binary_preds = float_to_binary(preds_squeezed, thresh=self.cfg.train.target_threshold)
        self.test_MAE(preds_squeezed, target_squeezed)
        self.test_MAE_OS(*get_outliers_s(preds_squeezed, target, thresh=self.cfg.train.target_threshold))                 
                
        # =========================== QUANTILE REGRESSION ============================                
        # binary_target = float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold)
        # score_preds = float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold)
        # binary_preds = float_to_binary(predictions[:, 0], thresh=self.cfg.train.target_threshold)
        # self.test_MAE(predictions[:, 0], target[:, 0])
        # self.test_MAE_full(predictions, target) # QUANTILE REGRESSION
        # self.test_MAE_OS(*get_outliers_s(predictions[:, 0], target, thresh=self.cfg.train.target_threshold)) # Strange, why tagret, not target[:, 0]? 13.09.25 
        # =========================== QUANTILE REGRESSION ============================        
        self.test_AP(score_preds, binary_target)
        self.test_precision(binary_preds,binary_target)
        self.test_recall(binary_preds, binary_target)
        self.test_auroc(score_preds, binary_target)
        
        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/MAE", self.test_MAE, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("test/MAE_full", self.test_MAE_full, on_step=True, on_epoch=True, prog_bar=False) # QUANTILE REGRESSION

        self.log("test/MAE_OS", self.test_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/AP", self.test_AP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/AUROC", self.test_auroc, on_epoch=True)

        # if batch_idx%100==0:
        #     self.log("test/target_96", target[:, 0])
        #     self.log("test/prediction_96", predictions[:, 0])
        #     self.log("test/target_50", target[:, 3])
        #     self.log("test/prediction_50", predictions[:, 3])
        #     self.log("test/target_05", target[:, 6])
        #     self.log("test/prediction_05", predictions[:, 6])
            # self.logger.experiment.log({"test/target_96": target[:, 0], "test/prediction_96": predictions[:, 0]})
            # self.logger.experiment.log({"test/target_50": target[:, 3], "test/prediction_50": predictions[:, 3]})
            # self.logger.experiment.log({"test/target_05": target[:, 6], "test/prediction_05": predictions[:, 6]})
        
        output = OrderedDict(
            {
                "loss": loss,
                "binary_preds": binary_preds,
                "score_preds": score_preds,
                "float_preds": predictions,
                "float_target": target,
                "binary_target": binary_target,
            }
        )
        self.test_outputs.append(output)

        return output
    
    def on_test_start(self):
        print(f'Plots will be saved to {self.run_dir}')
        self.test_outputs = []

    def on_test_epoch_end(self):
        preds = torch.stack([x["binary_preds"] for x in self.test_outputs]).to(dtype=torch.float32).cpu().flatten()
        target = torch.stack([x["binary_target"] for x in self.test_outputs]).to(dtype=torch.int32).cpu().flatten()
        preds_float = torch.stack([x["float_preds"] for x in self.test_outputs]).to(dtype=torch.float32).cpu().flatten()
        target_float = torch.stack([x["float_target"] for x in self.test_outputs]).to(dtype=torch.int32).cpu().flatten()
        print(f"В on_test_epoch_end() -> preds_float.shape: {preds_float.shape}, target_float.shape: {target_float.shape}")
        preds_float = preds_float.squeeze() # Comment for Quantile Regression квантильная регрессия
        # <<< НАЧАЛО БЛОКА ДЛЯ СЧЕТЧИКА >>>

        # 1. Берем порог из конфига
        threshold = self.cfg.train.target_threshold

        # 2. Считаем, сколько значений в target_float больше этого порога
        outlier_count = torch.sum(target_float > threshold).item()
        total_samples = len(target_float)

        # 3. Выводим информацию в консоль
        print(f"\n--- Статистика по выбросам (скорость > {threshold} м/с) ---")
        print(f"Количество случаев: {outlier_count} из {total_samples} ({outlier_count / total_samples:.2%})")
        print("--------------------------------------------------")

        # <<< КОНЕЦ БЛОКА ДЛЯ СЧЕТЧИКА >>>

        # <<< НАЧАЛО БЛОКА, КОТОРЫЙ НУЖНО ДОБАВИТЬ >>>

        print("\n--- Анализ результатов по бинам (тестовый набор) ---")
        # Вызываем вашу функцию с предсказаниями и реальными значениями
        binned_results_df = analyze_performance_by_bins(
            y_pred=preds_float.numpy(), 
            y_true=target_float.numpy(),
            n_bins=5
        )
        print(binned_results_df)
        print("--------------------------------------------------")

        # (Очень рекомендуется) Сохраняем эту таблицу в CSV и логируем в MLflow как артефакт
        binned_results_path = os.path.join(self.run_dir, 'binned_test_results.csv')
        binned_results_df.to_csv(binned_results_path)
        self.logger.experiment.log_artifact(binned_results_path)

        # <<< КОНЕЦ БЛОКА, КОТОРЫЙ НУЖНО ДОБАВИТЬ >>>
        thrs = [0, 3, 5, 8, 10, 12, 15, 17, 20, 23, 25, 27, 30]
        rmses = []
        for th in thrs:
            if len(torch.where(target_float >= th)[0])>0:
                tgt_th = target_float[torch.where(target_float >= th)[0]]
                pred_th = preds_float[torch.where(target_float >= th)[0]]
                rmses.append(np.squeeze(torch.sqrt(torch.mean((pred_th - tgt_th) ** 2)).numpy()))
        
        fig, ax = plt.subplots()
        ax.plot(thrs, rmses, color='purple')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Wind Speed (m/s)')
        fig.savefig(os.path.join(self.run_dir, 'RMSE_vs_target.png'))   # save the figure to file        

        precision, recall, thresholds = precision_recall_curve(target, preds)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        fig.savefig(os.path.join(self.run_dir, 'PR_curve.png'))   # save the figure to file     
        
        # 1. Scatter plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.scatter(target_float.numpy(), preds_float.numpy(), alpha=0.1)
        ax_scatter.plot([target_float.min(), target_float.max()], [target_float.min(), target_float.max()], 'r--', lw=2) # Диагональ y=x
        ax_scatter.set_xlabel('Истинные значения (м/с)')
        ax_scatter.set_ylabel('Предсказанные значения (м/с)')
        ax_scatter.set_title('Предсказание vs. Истина')
        ax_scatter.grid(True)
        # Сохраняем в MLflow
        self.logger.experiment.log_figure(fig_scatter, "test_scatter_plot.png")

        # 2. Гистограмма ошибок
        errors = (preds_float - target_float).numpy()
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(errors, bins=50)
        ax_hist.set_xlabel('Ошибка предсказания (м/с)')
        ax_hist.set_ylabel('Частота')
        ax_hist.set_title('Распределение ошибок')
        self.logger.experiment.log_figure(fig_hist, "test_error_distribution.png")

        # 3. Сохранение сырых предсказаний для дальнейшего анализа
        # Это КРАЙНЕ ВАЖНО для воспроизводимости и статистических тестов
        results_df = pd.DataFrame({
            'prediction': preds_float.numpy(),
            'target': target_float.numpy()
        })
        results_df.to_csv(os.path.join(self.run_dir, 'test_predictions.csv'), index=False)
        self.logger.experiment.log_artifact(os.path.join(self.run_dir, 'test_predictions.csv'))   


    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(),
                                    lr=self.cfg.train.learning_rate,
                                    weight_decay=self.cfg.train.weight_decay)        
        if self.scheduler_name is not None:
            if self.scheduler_name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.7, patience=300, verbose=True, interval="step", frequency=1)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        "monitor": "train/loss",
                        "patience": 300,
                        "mode": "min",
                        "factor": 0.7,
                        "verbose": True,
                        'name': 'train/lr',
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }
            
            elif self.scheduler_name == "OneCycleLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'name': 'train/lr',  # put lr inside train group in tensorboard
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }
            elif self.scheduler_name == "LinearLR":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                            start_factor=1.0, end_factor=0.2, 
                                                            total_iters=self.trainer.estimated_stepping_batches)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'name': 'train/lr',
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }

        else:
            return optimizer
        
    def on_after_backward(self):
        # Проверяем градиенты только на первых двух шагах обучения
        if self.trainer.global_step % 100 == 0:
            print("\n" + "#"*50)
            print(f"--- ДЕБАГ ГРАДИЕНТОВ (ПОСЛЕ ШАГА {self.trainer.global_step}) ---")

            # Градиенты для весов первого Linear слоя в "голове"
            grad_lin1 = self.net.head_lin1.weight.grad
            if grad_lin1 is not None:
                print("\nГрадиенты для Linear_1 (до активации):")
                print(f"  📈 Среднее абсолютное значение градиента: {grad_lin1.abs().mean():.6f}")
                print(f"  📈 Максимальное абсолютное значение: {grad_lin1.abs().max():.6f}")
            else:
                print("\nГрадиенты для Linear_1 отсутствуют (None)!")

            # Градиенты для весов второго, финального Linear слоя
            grad_lin2 = self.net.head_lin2.weight.grad
            if grad_lin2 is not None:
                print("\nГрадиенты для Linear_2 (финальный слой):")
                print(f"  📈 Среднее абсолютное значение градиента: {grad_lin2.abs().mean():.6f}")
                print(f"  📈 Максимальное абсолютное значение: {grad_lin2.abs().max():.6f}")
            else:
                print("\nГрадиенты для Linear_2 отсутствуют (None)!")
            
            print("#"*50 + "\n")
            
def analyze_performance_by_bins(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 5):
    """
    Анализирует производительность модели, разбивая тестовые данные на бины
    по значению целевой переменной, и считает метрики для каждого бина.
    """
    
    # 1. Создаем DataFrame для удобства работы
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })

    # 2. Разбиваем весь диапазон целевых значений на n_bins равных интервалов (бинов).
    # Например, если y_true от 0 до 50, и n_bins=5, то бины будут [0-10), [10-20), ..., [40-50].
    bin_edges = [0, 5, 10, 15, 20, 25, 30, np.inf]
    bin_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "> 30"]
    df['bin'] = pd.cut(df['y_true'], bins=bin_edges, labels=bin_labels, right=False)
    # right=False означает, что интервал включает левую границу: [8, 15), [15, 20)
    
    # 3. Считаем, сколько примеров попало в каждый бин, чтобы определить их "редкость"
    bin_counts = df.groupby('bin').size()
    
    # 4. Ранжируем бины: Rank 1 - самый редкий, Rank 5 - самый частый
    bin_ranks = bin_counts.rank(method='first').astype(int)

    # <<< НАЧАЛО БЛОКА ДЛЯ ПЕЧАТИ РЕДКИХ СЛУЧАЕВ >>>

    # Находим имя самого редкого бина (где ранг равен 1)
    # .idxmax() на инвертированных рангах найдет индекс минимального значения
    rarest_bin_name = bin_ranks.idxmin() 
    
    # Фильтруем DataFrame, чтобы получить только строки, относящиеся к этому бину
    rarest_samples_df = df[df['bin'] == rarest_bin_name]

    print(f"\n--- Детальный разбор самого редкого бина: '{rarest_bin_name}' ---")
    # Округляем значения для наглядности и печатаем
    print(rarest_samples_df.round(2))
    
    # <<< КОНЕЦ БЛОКА ДЛЯ ПЕЧАТИ РЕДКИХ СЛУЧАЕВ >>>

    # 5. Считаем метрики для каждого бина
    def calculate_rmse(group):
        # Если группа (бин) пустая, возвращаем NaN, иначе считаем метрику
        if group.empty:
            return np.nan
        return np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
    
    def calculate_mae(group):
        if group.empty:
            return np.nan
        return mean_absolute_error(group['y_true'], group['y_pred'])      
    
    bin_metrics = df.groupby('bin', observed=False).apply(lambda x: pd.Series({
        'RMSE': calculate_rmse(x),
        'MAE': calculate_mae(x)
    }))

    # 6. Собираем всё в красивую итоговую таблицу
    results_df = pd.DataFrame({
        'Bin Rank': bin_ranks,
        'Sample Count': bin_counts,
    }).join(bin_metrics).reset_index()

    # Сортируем по рангу для наглядности
    results_df = results_df.sort_values(by='Bin Rank').set_index('Bin Rank')
    
    return results_df