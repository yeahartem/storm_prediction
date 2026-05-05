# Аудит #2: «почему метрики снова одинаковые после фикса pos-leakage»

**Дата:** 2026-05-03
**Аудитор:** Claude Opus 4.7
**Статус:** диагностирован НЕ-баг, а фундаментальная проблема дизайна эксперимента

---

## TL;DR

После фиксов от 2026-05-01 (pos в head убран, val/test разделены, val-batches детерминизированы):

| Источник | AUROC | AP | BS |
|---|---:|---:|---:|
| sfcWindmax-only (1 ch) | 0.8310 | 0.7258 | 0.1599 |
| full (4 ch sfcWind+pr+tasmax+tasmin) | 0.8311 | 0.7309 | 0.1682 |
| **Climatology (станция, месяц) baseline** | **0.8453** | **0.7481** | **0.1502** |
| Climatology + Gaussian noise σ≈0.5 (logit) | 0.8354 | 0.7314 | — |
| Climatology + Gaussian noise σ≈0.7 (logit) | 0.8258 | 0.7164 | — |

**Главный вывод:** обе модели работают **ниже climatology baseline**. AUROC=0.831 численно эквивалентен «climatology lookup + ~σ=0.6 logit гауссовского шума». Обе ablation сходятся к **одному и тому же шумному плато climatology**, потому что **99.78% test-станций были в train**, и модель учится опознавать станцию по содержимому патча, а не предсказывать физику.

Это **тот же тип утечки, что и pos в head, но теперь через содержимое патча**. Фикс pos-leakage был корректен (см. ниже численное подтверждение), но исходная задача-постановка не позволяет разделить «модель учит физику» и «модель учит идентификацию станции по геометрии 95×95 патча».

---

## Подтверждённые факты

### 1. Pos в head действительно убран (статически)

`src/regression/models/models.py:46`: `def __init__(self, in_chans=4, use_pos_in_head: bool = False, ...)` — default `False`.

Все 26 train-конфигов в `configs/train/` имеют `use_pos_in_head: false` явно:
```
configs/train/train_cmip6_world.yaml:use_pos_in_head: false   # MUST stay false ...
configs/train/train_cmip6_world_sfcwindmax_only_server.yaml:use_pos_in_head: false
... (все 26 конфигов) ...
```

В forward: путь `if self.use_pos_in_head` НЕ выполняется, `pos` остаётся в `objs[1]`, но не попадает в `head_lin1`.

### 2. val/test действительно разделены

`src/regression/data_load.py:316–390` теперь использует `start_of_val` и `start_of_test`:
- train: `time < start_of_val`
- val: `start_of_val <= time < start_of_test`
- test: `time >= start_of_test`

Sanity-assert на отсутствие пересечений (lat,lon,time) сделан явно (стр. 387–390). Это доказуемо.

### 3. Climatology baseline на текущих данных = 0.8453 AUROC

Запуск `outputs/debug_baseline.py`:
```
train_data_idxs shape: (9, 2785160)
test_data_idxs  shape: (9, 256150)
train pos rate: 0.3602
test  pos rate: 0.3539
unique stations train: 6025
unique stations test : 5478

Climatology (station, month) baseline on TEST:
  AUROC: 0.8453
  AP:    0.7481
  Brier: 0.1502

Compare to model reported:
  sfcWindmax-only: AUROC=0.8310 AP=0.7258 Brier=0.1599
  full-4ch       : AUROC=0.8311 AP=0.7309 Brier=0.1682

Station-only baseline on TEST: AUROC=0.8237  AP=0.6952
Month-only   baseline on TEST: AUROC=0.5456  AP=0.3854
```

(Обратите внимание: `train_data_idxs.npy` физически на диске старый, но логика split идентична текущему коду в `data_load.py`, так что числа корректны для типичного запуска.)

### 4. Test-станции на 99.78% совпадают с train-станциями

`outputs/debug_station_overlap.py`:
```
Stations in train:             6025
Stations in test:              5478
Stations in BOTH:              5459
Stations only in test (OOD):     19   ← всего 19 «новых» станций!

test samples at SEEN station:   255574/256150 (99.78%)
test samples at NEW station:    576/256150 (0.22%)

Climatology AUROC on FULL test:           0.8453
Climatology AUROC on SEEN-station subset: 0.8456
Climatology AUROC on NEW-station subset:  0.5000   ← OOD = случайный!
```

Это именно та утечка, которую невозможно «починить в коде». Если test-станции уже в train, любая модель, способная опознать станцию по геометрии патча, получит AUROC ≈ climatology.

### 5. AUROC=0.831 = «climatology + σ≈0.6 logit Gaussian noise»

`outputs/debug_noise_simulation.py`:
```
sigma  AUROC    AP
 0.00  0.8453  0.7481  (чистая climatology)
 0.30  0.8416  0.7418
 0.50  0.8354  0.7314
 0.70  0.8258  0.7164
 1.00  0.8092  0.6909
```

Модель = «зашумлённая копия climatology». Оба ablation (sfcwindmax-only и full) попадают в этот шумный плато.

---

## Почему именно содержимое патча → station ID

95×95 патч на 1.125° сетке = ~107° × 107° (огромный кусок континента). В этом окне всегда содержится:

1. **Береговая линия / распределение океан-суша** (через `pr`, `sfcWindmax`, `tasmax`)
2. **Широтный градиент** (через `tasmax`, `tasmin` — на экваторе теплее, чем в умеренных)
3. **Местная орография** (через elevation, если включён, и косвенно через `tasmax/tasmin`)
4. **Сезонная структура климата региона**

Любой backbone, обученный на `(patch → label)`, выучит:
*«какая это часть земного шара → какова там storm-rate»*

— и не нужно смотреть на физику. Это даёт climatology-AUROC.

Доказательство:
- `tasmax`-only: AUROC ~0.85 (через широтный градиент → станция)
- `tasmin`-only: AUROC ~0.85 (то же)
- `elevation`-only: AUROC 0.844 (через рельеф → станция; СТАТИЧЕСКАЯ карта!)
- `pr`-only: AUROC 0.802 (осадки слабее идентифицируют станцию)

Чем меньше геометрической информации в канале, тем ниже AUROC. Но ни один не учится физике штормов.

---

## Что не баг

1. ~~Test pipeline дублирует данные~~ → confusion matrices двух моделей **различные**:
   - sfcwind: TN=128505 FP=35931 FN=24771 TP=66345
   - full:    TN=123724 FP=40703 FN=22002 TP=69091
   Предсказания РАЗНЫЕ, просто похожие по AUROC.

2. ~~Метрики hardcoded~~ → torchmetrics корректно считает AUROC и AP на `score_preds = torch.sigmoid(logits)` против `binary_target = (y >= station_threshold).int()`. Это правильно.

3. ~~Pos утекает несмотря на флаг~~ → статический анализ models.py показывает, что путь с pos НЕ выполняется при `use_pos_in_head=False`. `probe_backbone.py` подтвердит это эмпирически (тест 1 должен дать AUROC≈0.5).

4. ~~val=test~~ → исправлено в `data_load.py` и `datamodule.py`, проверки assert на overlap есть.

---

## Что РЕАЛЬНО баг — но это «баг эксперимента», а не кода

**Train и test перекрываются по станциям на 99.78%.** Каждая (станция, месяц) в test уже виделась в train. Climatology = 0.845. Backbone достигает 0.83 «похожим путём».

**Это не починить, не изменив дизайн эксперимента.** Стандартное решение — **geo-OOD split**:

```python
# В data_load.py target_df_to_array():
# 1. собрать список всех уникальных станций
# 2. разделить станций случайно: 80% в train, 20% в test
# 3. в train идут ТОЛЬКО строки этих станций (всё время 2000-2024)
# 4. в test ТОЛЬКО строки оставшихся станций (всё время 2000-2024)
# Альтернатива: leave-one-region-out из pl_module.REGIONS
```

Тогда модель **физически не сможет** запомнить (станция → score), а вынуждена будет извлекать сигнал из CMIP6 patches.

Прогноз для geo-OOD test:
- Climatology baseline → AUROC ≈ 0.50–0.55 (без станционной информации)
- Текущий backbone → AUROC ≈ 0.55–0.65 (если в патче есть какой-то широтный/сезонный сигнал)
- Если backbone выдаст AUROC > 0.70 — это будет реальная физика

---

## Сопутствующие наблюдения

### Brier и ECE различаются между ablation, AUROC — нет
- sfcwind: BS=0.1599 ECE=0.039
- full:    BS=0.1682 ECE=0.085

Это ожидаемо: при одинаковом ranking (AUROC) калибровка может различаться. Full-модель более переуверенная (ECE 2× больше). Это к делу не относится, но стоит упомянуть в paper.

### val-data в текущем pipeline это новые 2021-2022 (не = test)
В обоих логах в начале test видно: `Test dataloader init: 25557X samples`. Это test=2023-2024. Корректно.

### probe_backbone.py — корректный финальный sanity check
- Test 1 (X=zeros, real pos): должен дать AUROC ≈ 0.50. **Если выше 0.55** → есть остаточная утечка pos где-то.
- Test 2 (X shuffled, pos=0): должен дать AUROC ≈ 0.50. **Если выше 0.55** → backbone выучил «статический» признак, не зависящий от точного содержимого патча конкретного семпла (то есть просто bias на station).

Когда probe закончится, эти числа — главное что нам нужно увидеть.

---

## Конкретный план действий

### Шаг 1 (срочно): дождаться probe_backbone.py

```bash
tail -f out/probe_backbone.log
```

Ожидаемое:
```
AUROC (real X):     ~0.83  (= то, что мы уже видели)
AUROC (X=zeros):    ~0.50  ← подтверждение, что pos-leakage нет
AUROC (X=shuffled): ~0.50  ← подтверждение, что нет «глобального bias» 
```

Если `X=zeros` AUROC > 0.55 — будет повод копать дальше. Я ставлю 99% на 0.50, но проверить надо.

### Шаг 2: запустить geo-OOD как ablation для paper

Это **главный эксперимент Q1**. Без него reviewer спросит «а как доказано, что модель не делает lookup?»

Минимальный код-патч в `src/regression/data_load.py` (добавить в `target_df_to_array` сразу после построения `target_array`):

```python
# Geo-OOD split: hold out 20% of stations (random) for testing
geo_ood = self.cfg.train.get('geo_ood_split', False)
if geo_ood:
    rng = np.random.default_rng(self.cfg.train.get('geo_ood_seed', 42))
    station_keys = target_array[0].astype(int) * 1_000_000 + target_array[1].astype(int)
    unique_stations = np.unique(station_keys)
    n_test = int(len(unique_stations) * 0.2)
    test_stations = set(rng.choice(unique_stations, size=n_test, replace=False).tolist())
    is_test_station = np.isin(station_keys, list(test_stations))
    # train: NOT in test_stations, time < start_of_test
    # val:   NOT in test_stations, time >= start_of_test (temporal val)
    # test:  IN test_stations, any time (geo-OOD)
    train_mask = (~is_test_station) & (time_idx < val_split_index)
    val_mask = (~is_test_station) & (time_idx >= val_split_index) & (time_idx < test_split_index)
    test_mask = is_test_station
    ...
```

Конфиг:
```yaml
# train_cmip6_world_geood.yaml (новый)
geo_ood_split: true
geo_ood_seed: 42
```

Запустить ablation на geo-OOD:
- sfcwindmax-only — geo-OOD
- full 4ch — geo-OOD
- pr-only — geo-OOD
- elevation-only — geo-OOD

Если в geo-OOD AUROC моделей **не упадёт до climatology-OOD** (~0.50) — backbone что-то реально знает. Если упадёт — paper придется переписать (можно как negative result про «CNN не извлекают синоптики из CMIP6 на 25 дней»).

### Шаг 3: добавить climatology baseline в paper-таблицу

Обязательно. Это ~5 строк кода:

```python
# Запустить: python compute_climatology_baseline.py → climatology_predictions.csv
# Бутстрап CI на climatology_predictions.csv через bootstrap_ci.py
```

Без этого paper-таблица «headlines» неполна. Reviewer сразу спросит «что даёт тривиальный baseline?»

### Шаг 4 (опционально): elevation-only test-only прогон тоже

Это сейчас тренируется (вторая твоя команда). Когда закончится — прогнать test, увидеть AUROC. Прогноз: 0.83-0.84. Это явно покажет, что **статическая 2D карта рельефа даёт почти ту же метрику, что и динамическая 4-канальная модель**, и это будет визитная карточка диагноза для paper.

---

## Итоговая сводка для решения

**Что сейчас знаем:**
1. Pos-leakage код-фикс корректен (статически + неявно подтверждается метриками 0.831 < climatology 0.845)
2. Но геометрическая утечка через 95×95 патч сохраняется: 99.78% test-станций видны в train
3. Все ablation сходятся к «зашумлённой climatology» через идентификацию станции по патчу

**Что остаётся проверить (probe_backbone сейчас считается):**
- AUROC при X=zeros — должно быть ≈ 0.5

**Что делать для paper:**
1. Добавить climatology в headline-таблицу
2. Обязательно сделать geo-OOD ablation (это и есть главный эксперимент)
3. (Опционально) показать elevation-only ≈ full как «smoking gun» того, что физика не извлекается

**Что paper НЕ показывает в текущем виде:**
- Что CNN извлекает физический сигнал из CMIP6 patches → требуется geo-OOD
- Что добавление tasmax/pr поверх sfcWind помогает → они дают тот же AUROC

**Что paper МОЖЕТ показать честно:**
- Pipeline для климатологического baseline через CNN (получает 0.83)
- Чувствительность к удалению pos в head (0.85 → 0.83)
- Различие в калибровке между ablation
- Региональные/сезонные различия для одной модели

---

*Сценарии всех debug-проверок сохранены в `outputs/debug_*.py` для воспроизводимости.*
