# Аудит pipeline и поиск бага «одинаковые метрики»

**Дата:** 2026-05-01  
**Аудитор:** Claude Opus 4.6  
**Статус:** баг найден и подтверждён численно

---

## TL;DR

Главный баг — **information leakage через `pos`**. Модель не использует CMIP6 patches вообще, она почти полностью обучается на per-(станция, месяц) климатологии, которая закодирована в 4 числах `pos = [time_pos, time_pos_m, lat_pos, lon_pos]`. Эти 4 числа конкатенируются с backbone features и подаются в head — этого достаточно для построения lookup-таблицы «(lat_idx, lon_idx, month) → storm rate».

Численное подтверждение: простой baseline «средняя частота положительного класса по (станция, месяц), посчитанная на train» даёт на test:

| Baseline | AUROC | AP |
|---|---:|---:|
| **Climatology (станция, месяц)** | **0.8452** | **0.7477** |
| Climatology (1° lat, 1° lon, месяц) | 0.8452 | 0.7477 |
| Climatology (только станция) | 0.8237 | 0.6953 |

Сравните с результатами ваших ablation:

| Ablation | AUROC | AP | Δ AUROC vs baseline | Δ AP vs baseline |
|---|---:|---:|---:|---:|
| Full (4ch) | 0.851 | 0.764 | **+0.006** | **+0.016** |
| sfcWindmax-only | 0.847 | 0.755 | +0.002 | +0.007 |
| tasmax-only | 0.847 | 0.755 | +0.002 | +0.007 |
| tasmin-only | 0.849 | 0.756 | +0.004 | +0.008 |
| **elevation-only (статика!)** | 0.844 | 0.746 | **−0.001** | **−0.002** |
| pr-only | 0.802 | 0.689 | −0.043 | −0.059 |

При bootstrap CI ±0.002 AUROC / ±0.003 AP **все ablation-эксперименты** (кроме pr-only) находятся в пределах одного-двух CI от тривиального climatology baseline. CMIP6 backbone почти ничего не добавляет — реальный физический сигнал составляет ≤0.006 AUROC и ≤0.016 AP.

Случай **elevation-only** (статическая 2D карта высот, повторённая 27 раз во времени) выдаёт AUROC 0.844, что **на 0.001 ниже** climatology baseline на 4 числах. Это прямое доказательство, что backbone не используется — статическая карта работает в пределах шума так же, как полная физика.

Случай **pr-only** даёт 0.802 — на 0.043 НИЖЕ baseline. Это значит, что осадки сами по себе не несут сигнала, и при этом шумная переменная мешает оптимизации pos-пути.

---

## Где именно «течёт» информация

### `src/regression/datamodule.py` строки 142–143 и 190–191

```python
pos = torch.tensor([time_pos, time_pos_m, lat_pos, lon_pos], dtype=self.dtype)
pos = pos.expand(self.cfg.time_window, 4)   # 27 копий одних и тех же 4 чисел
```

То есть на одно образец — 4 уникальных числа: month/12, (month·30.5+day)/365, lat°/90, lon°/180. **Это полная сетка станции и календаря.**

### `src/regression/models/models.py` строки 51–67

```python
def forward(self, X):
    X, pos = X
    b = X.shape[0]
    days = X.shape[2]                                    # = 27
    X = torch.reshape(X, [b * days, X.shape[1], 95, 95])
    X = self.ghostnetv2(X)                               # (B*27, 70)
    pos = torch.reshape(pos, [b * days, 4])              # (B*27, 4)
    X = torch.cat((X, pos), 1)                           # (B*27, 74)
    X = torch.reshape(X, [b, days * (self.embed + 4)])   # (B, 27*74) = (B, 1998)
    X = self.head_dropout(X)
    X = self.head_lin1(X)                                # 1998 -> 70
    X = self.head_activation(X)
    X = self.head_lin2(X)                                # 70 -> 1
    return X
```

Что получает `head_lin1`:
- 27 × 70 = **1890** выходов backbone (зашумлённые, со случайной инициализацией)
- 27 × 4 = **108** входов от pos, **из которых только 4 уникальных** значения, повторённых 27 раз

`head_lin1` — это `Linear(1998, 70)`. Каждый из 70 выходов — это сумма 1998 взвешенных входов. Поскольку 108 pos-входов это просто 27 копий одного и того же вектора, эффективно head_lin1 имеет **прямой доступ к 4 pos-числам через сумму 27 индивидуальных весов** на каждое число. То есть для каждого выхода k:

```
h_k = bias + Σ(i,j) W_lin1[k, j+74·i] · backbone[i, j]
            + Σ(i,p) W_lin1[k, 70+p+74·i] · pos[p]
            = ... + Σ_p (Σ_i W_lin1[k, 70+p+74·i]) · pos[p]
```

Сумма 27 весов на каждое из 4 pos-чисел — это эффективно один скалярный коэффициент на pos[p]. Итого head_lin1 → 4 pos-входа × 70 выходов = 280 степеней свободы только на pos. Этого с лихвой хватает, чтобы построить нелинейный лукап (lat, lon, month) → climatology.

Backbone выдаёт случайно инициализированные шумные эмбеддинги. Градиент течёт по обоим путям, но pos сразу даёт сильный сигнал, потому что:
1. Это аналитически точная декомпозиция таргета (climatology)
2. Backbone требует длинного обучения, чтобы вычислять что-то осмысленное из 95×95 patches
3. С `Dropout(0.4)` на 27 копиях pos в среднем выживает ≈16 копий — больше чем достаточно

### `src/regression/data_load.py` строки 432–440 (генерация pos в `target_array`)

```python
target_array = np.stack([
    np.full(len(dates), lat),                  # row 0: lat_index (paddded)
    np.full(len(dates), lon),                  # row 1: lon_index (paddded)
    dates,                                     # row 2: time_index
    time_positions,                            # row 3: time_pos = (m*30.5+d)/365
    time_positions_m,                          # row 4: month/12
    np.full(len(dates), lat_position),         # row 5: lat°/90
    np.full(len(dates), lon_position),         # row 6: lon°/180
    y_max,                                     # row 7: max(wind, 28d window)
])
```

`pos` строится из rows 3–6, и эти значения **постоянны в пределах одной станции для всего времени.** Поэтому `pos` фактически уникально идентифицирует станцию (при ~1° сетке lat/lon).

---

## Почему все эксперименты дают одинаковый AUROC

1. Из `pos`-пути модель получает **per-(станция, месяц) climatology**, AUROC = 0.845
2. Backbone CNN добавляет ≤0.006 AUROC даже при идеальных входах
3. Bootstrap CI ±0.002 не позволяет различать ablation выше уровня climatology
4. Поэтому все эксперименты «выглядят как одна и та же модель» — потому что они и есть одна и та же модель «pos-only climatology + малый шумный довесок от backbone»

### Почему pr-only хуже?
- `pr` (осадки) центрируются около 0 и сильно правосторонне-смещены
- Дельта осадков плохо коррелирует с штормами в окне 28 дней
- Но `pr` **активно мешает** оптимизации: backbone тратит градиенты на бессмысленные patches, что косвенно увеличивает шум на head_lin1, и оптимизатор хуже использует pos
- Результат: pos-путь не доходит до 0.845, остаётся на 0.802

### Почему elevation-only ≈ climatology baseline?
- Статическая карта высот = идентичность станции через её ландшафт
- Backbone выдаёт примерно одинаковый embedding для каждого из 27 time-step (ведь патч одинаковый)
- Эти эмбеддинги — ещё один способ закодировать станцию, так же как pos
- AUROC = 0.844 ≈ AUROC climatology = 0.845

---

## Сопутствующие баги (вторичные, но усиливают эффект)

### 1. val == test (`src/regression/datamodule.py` строки 36–40)

```python
if stage == "fit" or stage is None:
    self.dataset_train = DatasetClass(DPL=self.DPL, test=False)
    self.dataset_val   = DatasetClass(DPL=self.DPL, test=True)   # ← test data!
if stage == "test":
    self.dataset_test  = DatasetClass(DPL=self.DPL, test=True)
```

`dataset_val` и `dataset_test` смотрят на одни и те же `DPL.test_data_idxs` (2023–2024). 2021–2022 попадает в train. Значит:
- Early stopping выбирает чекпоинт по test-метрикам
- Сообщаемый test AUROC — это AUROC чекпоинта, выбранного на тех же данных
- Это **вторая утечка**, систематически завышающая все метрики на одинаковую величину
- **Это не объясняет, почему ablation одинаковые** (ordering сохраняется), но добавляет к завышению

Чтобы исправить: создайте честный val split, например 2021-01-01 — 2022-12-31:

```python
val_start = self.time_coords.searchsorted(np.datetime64('2021-01-01'))
val_end   = self.time_coords.searchsorted(np.datetime64('2023-01-01'))
val_idxs  = train_array[:, (train_array[2] >= val_start) & (train_array[2] < val_end)]
train_idxs_only = train_array[:, train_array[2] < val_start]
```

### 2. `shuffle=True` в val_dataloader (`src/regression/datamodule.py` строка 56)

```python
def val_dataloader(self):
    return DataLoader(..., shuffle=True, ...)
```

С `limit_val_batches=1000` каждая эпоха валидируется на случайных 32K из 256K. Val/loss шумный → early stopping имеет случайную компоненту. В сочетании с val=test это означает, что финальный чекпоинт — это «удачное» совпадение random subset → test set.

### 3. `> split_index` вместо `>=` (`src/regression/data_load.py` строки 338–340)

```python
train_array = target_array[:, target_array[2, :] < split_index]
self.test_data_idxs = target_array[:, target_array[2, :] > split_index]
```

Один time-step ровно на 2023-01-01 теряется. Минор, но это типичный off-by-one и стоит пофиксить на `>= split_index` (с проверкой бордера).

### 4. CLAUDE.md заявляет «train 2000-2020», но факт «train 2000-2022»

Это нужно либо отразить в paper, либо пофиксить разделение. Лично я рекомендую обновить разделение, чтобы val было 2021-2022, test 2023-2024 (как и подразумевается в CLAUDE.md).

### 5. `use_pos_weight: false` при дисбалансе ~3:1

Это не баг, но важно: при positive rate 36% (train) bias of last layer уже учится правильно через BCEWithLogitsLoss. Однако без pos_weight модель оптимизирована на «балансе классов согласно prior», что для редких событий даёт менее острый PR. С `use_pos_weight: true` precision / recall были бы сбалансированы по-другому. Это оставлено как опция — оставьте `false`, если ваш дизайн так предполагает.

---

## Как доказать руками (за 5 минут)

Я уже сделал это и оставил скрипты в `outputs/`. Конкретно `probe3.py` строит per-(станция, месяц) lookup-таблицу на train и тестирует её на test. Результат:

```
Climatology (station, month) baseline:
  AUROC: 0.8452
  AP:    0.7477
  test samples in seen (st,mon) cells: 255020/256150 (99.6%)
```

99.6% test-семплов попадают в (станция, месяц)-ячейки, которые уже были в train. То есть «новых» (станция, месяц)-комбинаций почти нет, и любой классификатор, видящий pos, получит этот результат.

Чтобы вы могли воспроизвести у себя:

```python
import numpy as np
DATA = 'data/cmip6_world'
tr = np.load(f'{DATA}/train_data_idxs.npy')
te = np.load(f'{DATA}/test_data_idxs.npy')

y_tr = (tr[7] >= tr[8]).astype(np.float32)
y_te = (te[7] >= te[8]).astype(np.float32)

# (станция, месяц) lookup
key_tr = tr[0].astype(int)*10000 + tr[1].astype(int)
m_tr   = (tr[4]*12).round().astype(int)
ksm_tr = key_tr*100 + m_tr

uniq, inv = np.unique(ksm_tr, return_inverse=True)
mean = np.bincount(inv, weights=y_tr) / np.bincount(inv)
lookup = dict(zip(uniq, mean))

key_te = te[0].astype(int)*10000 + te[1].astype(int)
m_te   = (te[4]*12).round().astype(int)
ksm_te = key_te*100 + m_te
default = float(y_tr.mean())
pred = np.array([lookup.get(k, default) for k in ksm_te])

from sklearn.metrics import roc_auc_score, average_precision_score
print('AUROC:', roc_auc_score(y_te, pred))   # ≈ 0.845
print('AP:   ', average_precision_score(y_te, pred))  # ≈ 0.748
```

---

## Решающий тест на чекпоинте

Чтобы подтвердить, что работающая модель именно игнорирует backbone, прогоните любой обученный чекпоинт с `X = torch.zeros_like(X)` (только pos остаётся настоящим):

```python
# В test_step или в eval-скрипте
X, pos = batch[0]
X_zero = torch.zeros_like(X)
preds_zero = model([X_zero, pos])           # модель видит только pos
preds_real = model([X, pos])                # модель видит всё
```

Гипотеза: AUROC(preds_zero) ≈ AUROC(preds_real) — где-то 0.83–0.84. Если это подтвердится, backbone доказанно бесполезен.

---

## Что делать (рекомендации в порядке приоритета)

### 1. Уберите pos из head, либо тестируйте честно «pos-only» как baseline в paper

Самый простой способ — закомментировать конкатенацию:

```python
# X = torch.cat((X, pos), 1)        # ← убрать
X = torch.reshape(X, [b, days * self.embed])  # 27*70 = 1890
self.head_lin1 = nn.Linear(self.time_window * self.embed, 70)
```

Альтернатива — оставить pos, но прибавить **отдельный pos-only baseline** в таблице ablation в paper. Это честно показывает, что pos несёт 0.845 AUROC, и тогда «прирост от CMIP6» виден как реальные 0.006.

### 2. Введите real validation split

```python
# В data_load.py target_df_to_array()
val_start_date  = datetime.strptime('2021-01-01', '%Y-%m-%d').date()
test_start_date = datetime.strptime('2023-01-01', '%Y-%m-%d').date()
val_start  = self.time_coords.searchsorted(val_start_date)
test_start = self.time_coords.searchsorted(test_start_date)

mask_train = target_array[2] < val_start
mask_val   = (target_array[2] >= val_start) & (target_array[2] < test_start)
mask_test  = target_array[2] >= test_start

self.train_data_idxs = target_array[:, mask_train]
self.val_data_idxs   = target_array[:, mask_val]
self.test_data_idxs  = target_array[:, mask_test]
```

И в `datamodule.py`:
```python
self.dataset_val = DatasetClass(DPL=self.DPL, split='val')
```

### 3. Уберите `shuffle=True` из val_dataloader или перестаньте использовать `limit_val_batches`

Случайное подмножество для валидации делает чекпоинт-селекцию шумной.

### 4. Покажите анти-leakage эксперимент в paper

Сделайте ablation, в котором вы:
- Используете тот же CMIP6 patches
- Но НЕ конкатенируете pos в head
- Берёте новый случайный split станций (geo-out validation), где станции, не виденные при обучении, идут в test

Тогда CMIP6 будет вынужден вытащить физический сигнал, без возможности lookup'а станции.

### 5. Сделайте geo-OOD сплит

Вместо temporal-only split (2000-2022 vs 2023-2024) сделайте дополнительно случайную выборку станций для held-out OOD-теста. Текущий test содержит на 99.6% те же станции, что и train, поэтому lookup тривиален. Если bootstrap дать модели увидеть только новые станции, будет понятно, насколько backbone реально работает.

---

## Ответы на ваши вопросы из FOR_OPUS_debug_similar_metrics.md

**Q1: Игнорирует ли модель backbone?**  
Да, практически полностью. Численно: ≤0.006 AUROC прирост к pos-only climatology. Подтверждено через эквивалентный baseline.

**Q2: Получают ли разные ablation одинаковые данные?**  
Нет, конфиги действительно разные (разные `variables`, `in_chans`, `use_elevation`). Но это неважно, потому что главный сигнал идёт по pos-каналу.

**Q3: Корректен ли расчёт loss/метрик?**  
Да. `_score()` возвращает `torch.sigmoid()` для BCELoss, `_binary_pred()` возвращает `(logit > 0)`. AUROC и AP считаются через torchmetrics на `score_preds`, `binary_target = (y >= station_threshold).int()`. Логика чистая.

**Q4: `limit_val_batches=1000` + `shuffle=True` для val?**  
Да, это вторичный фактор шума, но **не основная причина**. Основная — pos leakage.

**Q5: Почему elevation-only даёт 0.844?**  
Потому что любой backbone, видящий статические per-станционные patches (или динамические), даёт примерно тот же per-station identity, что и pos. AUROC ≈ climatology baseline = 0.845. Backbone не открывает физику — он добавляет шумную копию станции-id, которая уже есть в pos.

---

## Вывод

**Это не баг в смысле «крашит метрику», это баг в смысле «pipeline не позволяет модели чему-то учиться сверх climatology».** Все ваши ablation честно (без двойного запуска и без идентичных данных) приходят к одному и тому же ответу: «(станция, месяц) → storm rate». Это естественный потолок при данной задаче и текущей структуре splits.

Чтобы paper имел научную ценность:
1. **Покажите pos-only baseline в таблице 8** (`AUROC=0.845`) — без этого все остальные строки выглядят почти одинаково
2. **Добавьте geo-OOD ablation** — тест на полностью невиданных станциях. Только в этом сценарии CMIP6 backbone сможет показать, есть ли в нём вообще физический сигнал
3. **Исправьте val=test** — иначе все цифры рискуют быть отвергнутыми reviewer'ом
4. **Подумайте, что вы реально хотите показать** — climatology lookup-table за минуту делается на CPU и выдаёт 0.845; backbone CNN с 11M параметрами и часами обучения добавляет 0.006. Если цель paper — продемонстрировать ценность CNN, текущий setup её НЕ демонстрирует
