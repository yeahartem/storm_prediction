# TODO: команды запустить + решения по статье

## КОМАНДЫ (запустить на сервере)

### 1. Узнать количество станций в датасете
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet')
print('Stations:', df['station_id'].nunique())
print('Countries:', df['country'].nunique() if 'country' in df.columns else 'no country col')
print('Date range:', df['date'].min(), '-', df['date'].max())
"
```
→ Вставить число в tex: `\textcolor{brown}{XXX}` в Data параграфе

### 2. Узнать итоговый % позитивных меток
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet')
# посмотреть структуру
print(df.columns.tolist())
print(df.head(3))
"
```
→ Уточнить "approximately 42%" в Definition of Target

### 3. После завершения обучения — финальные метрики для Table 2
Когда обучение закончится (ждём результатов):
- AUROC на тесте (глобальный)
- AP (Average Precision)
- F1, Precision, Recall при пороге 0.5
- Региональные метрики по 8 зонам (из regional_metrics.csv в MLflow)
- Сезонные метрики DJF/MAM/JJA/SON (из seasonal_metrics.csv)

### 4. Ablation study (no elevation)
После основного обучения запустить:
```bash
python train.py --config-name cmip6_world_noelev_server
```
→ Нужно для Table 3 (ablation): elevation vs no elevation

### 5. Violin plots нужно перегенерировать на CMIP6 данных
Старые pics/cmip_temperature.png и pics/cmip5_wind.png — из CMIP5.
Нужны новые с CMIP6 MRI-ESM2-0 за 2000-2022, глобально.
```bash
# запустить скрипт для violin plots (написать отдельно)
python make_violin_plots.py
```

---

## РЕШЕНИЯ (обсудить)

### Case Study: Ian и Amphan — оставлять?

**Проблема:** Hurricane Ian (сентябрь 2022) и Cyclone Amphan (май 2020) — оба попадают в **тренировочный период** (2000–2022). Модель "видела" эти данные → нечестная демонстрация.

**Варианты:**

**A) Взять штормы из тест-периода 2023–2024** ← рекомендую
- Честно: модель точно не видела
- Кандидаты: Cyclone Biparjoy (июнь 2023, Аравийское море/Индия), Storm Daniel (сентябрь 2023, Ливия/Средиземноморье), Typhoon Doksuri (июль 2023, Филиппины/Китай)
- Нужно: запустить inference на этих датах и посмотреть карты

**B) Оставить Ian/Amphan, честно написать "qualitative visualization"**
- Написать явно: "These cases are drawn from the training period and serve as a qualitative illustration..."
- Слабее для рецензентов, но быстрее

**C) Убрать Case Study совсем**
- Заменить на карту глобального storm probability на тестовом году (2023-2024)
- Чище, меньше работы
- Потеря "wow effect" от знаменитых ураганов

→ **Нужно решить: A, B или C?**

---

## ЧТО МЫ ДЕЛАЛИ В ЭКСПЕРИМЕНТЕ (для Methods секции)

Хронология ключевых решений которые вошли в финальный вариант:

### Данные
- **Источник**: CMIP6 MRI-ESM2-0, 4 переменные: sfcWindmax, pr, tasmax, tasmin
- **Станции**: GSOD глобальные, фильтр — минимум 25 валидных наблюдений за 6 месяцев
- **Временной диапазон**: 2000-01-01 — 2024-12-31
- **Train/Test split**: хронологический, train 2000–2022, test 2023–2024

### Целевая переменная (важное изменение!)
- Изначально была: p96 квантиль скользящего окна → регрессия
- **Финальная версия**: `max(wind_station, 28 days) >= max(q95_station, 15.0 m/s)` → бинарная классификация
- Обоснование: более чистая интерпретация ("был ли шторм в этот период"), лучше для публикации

### Архитектура
- GhostNetV2 backbone (timm) как spatial feature extractor
- Входной патч: 95×95 пикселей (half_side=47), 27 дней (time_window=27)
- 5 каналов: 4 CMIP6 переменных + elevation (статичный)
- Голова: Dropout → Linear(1998→70) → LeakyReLU → Linear(70→1)
- Позиционное кодирование: [sin(day/365), month/12, lat/90, lon/180] конкатенируется перед головой

### Обучение
- Loss: BCEWithLogitsLoss с pos_weight = neg/pos (борьба с дисбалансом классов)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: LinearLR (1.0 → 0.2 за все шаги)
- Precision: fp16 mixed (ускорение ~2x)
- Distributed: DDP, 2× RTX A5000 (24GB each)
- limit_train_batches=5000, batch_size=32 → 5000×32=160K примеров/эпоха
- Shuffle=True — критично для равномерного географического покрытия

### Метрики
- Основные: AUROC, Average Precision (AP), F1, Precision, Recall
- Региональные: 8 регионов (Russia Europe/W.Siberia/E.Siberia/Far East, Africa Equatorial/South/NE/Sahel)
- Сезонные: DJF/MAM/JJA/SON
- Confusion matrix на тестовом наборе

### Результат (промежуточный, epoch 18)
- val/AUROC = 0.848
- Лучший регион: russia_west_siberia AUROC=0.853
- Слабый: russia_east_siberia AP=0.29 (только 4.7% штормов — мало станций)
