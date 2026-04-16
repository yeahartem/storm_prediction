# Literature Review: Deep Learning for Extreme Wind Prediction from CMIP6

## Category 1: Statistical Downscaling / Bias Correction for Wind

**[1] Baño-Medina et al. 2022 (DeepESD) — GMD**
DOI: 10.5194/gmd-15-6747-2022
CNN downscaler for CMIP5 multi-model ensemble → EUR-44 grid. First DL contribution to CORDEX.
BENCHMARK: Yes — CNN downscaling baseline on climate model output.

**[2] Zhang et al. 2021 (BiGRU CMIP6 offshore wind) — Energy**
DOI: 10.1016/j.energy.2020.119321
BiGRU downscales CMIP6 near-surface wind to 0.25° for China offshore wind projections to 2100.
BENCHMARK: Partial — regression-based, not binary classification.

**[3] Buster et al. 2024 (SRDRN multivariate bias correction) — Climate Dynamics**
DOI: 10.1007/s00382-024-07406-9
Super-resolution deep residual network + quantile delta mapping for 6 CMIP6 variables including wind.
BENCHMARK: Yes — multivariate DL bias-correction baseline.

**[4] Harder et al. 2023 (Hard-Constrained DL) — JMLR / NeurIPS Workshop**
DOI: 10.5555/3648699.3649064
Physics-conservation constraints (renormalization layers) in CNN downscaling of ERA5 wind.
BENCHMARK: Partial — architectural reference.

**[5] Srivastava et al. 2023 (improved DL downscaling) — Heliyon**
DOI: 10.1016/j.heliyon.2023.e18596
Compares CNN, LSTM, ConvLSTM for downscaling wind; recurrent layers improve tail-event skill.
BENCHMARK: Partial — architecture comparison methodology.

**[6] Lian et al. 2024 (TerraWind) — GRL**
DOI: 10.1029/2024GL112124
Deep CNN for near-surface wind downscaling in complex terrain, explicit topographic effects.
BENCHMARK: No — terrain regression task.

**[7] Shen et al. 2022 (CMIP6 wind bias evaluation) — NYAS**
DOI: 10.1111/nyas.14910
Quantifies CMIP6 systematic wind biases globally; projects NH wind speed decreases to 2100.
BENCHMARK: No — motivation/diagnostic paper.

---

## Category 2: Extreme Wind and Storm Prediction with DL

**[8] Schulz & Lerch 2022 (DRN wind gusts) — MWR**
DOI: 10.1175/MWR-D-21-0150.1
Compares 8 ML methods for probabilistic wind gust forecasting at 175 German stations. DRN is strongest.
BENCHMARK: Yes — strongest ML wind postprocessing baseline in literature.

**[9] Rasp & Lerch 2018 (station embedding NNs) — MWR**
DOI: 10.1175/MWR-D-18-0187.1
NN with station embeddings for ensemble postprocessing; outperforms EMOS for temperature.
BENCHMARK: Yes — station-embedding neural network baseline.

**[10] Accarino et al. 2024 (WGPNet) — EACFM**
DOI: 10.1080/19942060.2024.2305318
CNN + multi-head attention predicts 2D wind gust fields from ERA5 at 0.25°.
BENCHMARK: Partial — regression, not binary, but highly relevant.

**[11] Outten & Sobolowski 2021 (CORDEX wind extremes) — Weather and Climate Extremes**
DOI: 10.1016/j.wace.2021.100363
15-member Euro-CORDEX ensemble + POT approach for future extreme wind projections over Europe.
BENCHMARK: Yes — POT/GEV baseline for same extreme wind projection task.

**[12] Kumar et al. 2015 (CMIP5 wind extremes) — Climate Dynamics**
DOI: 10.1007/s00382-014-2306-2
Evaluates CMIP5 models' ability to simulate annual maximum wind speeds vs ERA-Interim.
BENCHMARK: Yes — traditional climate model evaluation = non-ML baseline.

**[13] Michel et al. 2023 (QM bias correction wind Scandinavia) — JGR Atmospheres**
DOI: 10.1029/2022JD037953
Quantile mapping bias correction → EURO-CORDEX wind → return period changes over Scandinavia.
BENCHMARK: Yes — classical QM + extreme analysis pipeline.

---

## Category 3: CNN / DL on Gridded Climate Data

**[14] Reichstein et al. 2019 (DL Earth system review) — Nature**
DOI: 10.1038/s41586-019-0912-1
Landmark review: CNNs on spatiotemporal gridded data, need for hybrid physics-ML. >2000 citations.
BENCHMARK: No — framing paper.

**[15] Ham et al. 2019 (ENSO CNN) — Nature**
DOI: 10.1038/s41586-019-1559-7
Transfer learning: CNN trained on CMIP5 → reanalysis for ENSO prediction 1.5yr ahead.
BENCHMARK: No — forecasting, canonical methodological reference.

**[16] Kashinath/Prabhat et al. 2021 (ClimateNet) — GMD**
DOI: 10.5194/gmd-14-107-2021
Expert-labeled segmentation masks of tropical cyclones/ARs in CAM5.1; CGNet semantic segmentation.
BENCHMARK: Yes — pixel-level classification of extreme events from climate model grids.

**[17] Racah et al. 2017 (ExtremeWeather) — NeurIPS**
Multichannel spatiotemporal CNN for extreme weather pattern detection in CAM5.
BENCHMARK: Yes — foundational benchmark for extreme event classification from gridded climate data.

**[18] Rasp et al. 2020 (WeatherBench) — JAMES**
DOI: 10.1029/2020ms002203
ERA5-based benchmark for data-driven weather forecasting; CNN/ResNet baselines for wind speed. >500 citations.
BENCHMARK: Yes — CNN/ResNet baselines for gridded wind prediction.

**[19] Molina et al. 2021 (benchmark DL storms) — Earth and Space Science**
DOI: 10.1029/2020EA001490
CNN trained on WRF current-climate classifies severe storms; generalization to future climate tested.
BENCHMARK: Yes — benchmark methodology for climate generalization of DL classifiers.

**[20] Pathak et al. 2022 (FourCastNet) — arXiv / PASC 2023**
DOI: 10.1145/3592979.3593412
Fourier Neural Operators for global atmospheric forecasting including surface wind at 0.25°.
BENCHMARK: Partial — upper-bound context for wind prediction skill.

---

## Category 4: Benchmarks and Baselines

**[21] Sillmann et al. 2013 (CMIP5 ETCCDI indices) — JGR**
DOI: 10.1002/jgrd.50203
Evaluates CMIP5 on 27 ETCCDI extremes indices including percentile exceedance rates. >2500 citations.
BENCHMARK: Yes — canonical index-based extreme classification baseline.

**[22] Kharin et al. 2013 (CMIP5 GEV extremes) — Climatic Change**
DOI: 10.1007/s10584-013-0705-8
20-year return values for temperature/precipitation extremes in CMIP5 via GEV distributions.
BENCHMARK: Yes — GEV-based extreme probability estimation = standard non-ML baseline.

**[23] Dueben & Bauer 2018 (ML NWP challenges) — GMD**
DOI: 10.5194/gmd-11-3999-2018
ECMWF analysis of what it takes for ML to replace/supplement NWP; design principles.
BENCHMARK: No — foundational design-choices paper.

**[24] Schultz et al. 2021 (Can DL beat NWP?) — Phil Trans R Soc A**
DOI: 10.1098/rsta.2020.0097
Review: DL cannot yet replace NWP but strong for post-processing and extreme event detection.
BENCHMARK: No — review context paper.

**[25] Bochenek & Ustrnul 2022 (ML weather survey) — Atmosphere**
DOI: 10.3390/atmos13020180
Survey of ML for weather/climate including logistic regression and RF for extreme events.
BENCHMARK: Yes — logistic regression/RF baselines for extreme prediction.

---

## Category 5: CMIP6 for Impact Studies

**[26] Krishnan et al. 2020 (CMIP5/6 wind skill) — Climate Dynamics**
DOI: 10.1007/s00382-020-05406-z
CMIP6 improves over CMIP5 for Bay of Bengal winds but biases remain. Motivates adaptive threshold.
BENCHMARK: No — evaluation/motivation.

**[27] Lin et al. 2025 (CMIP6 typhoon wind) — ASCE**
DOI: 10.1061/AJRUA6.RUENG-1712
CMIP6 SST projections → typhoon track model → future extreme wind speeds.
BENCHMARK: No — application validating CMIP6-to-wind pipeline.

**[28] Soriano et al. 2023 (CMIP6 wind classification Spain) — JCP**
DOI: 10.1016/j.jclepro.2023.139742
Bias-corrected CMIP6 + percentile-based classification of offshore wind energy resource.
BENCHMARK: Yes — CMIP6-to-wind-classification pipeline directly comparable.

**[29] Zappa et al. 2024 (ML CORDEX-CMIP6 bridge) — arXiv**
ML emulator trained on CMIP5-CORDEX, applied to CMIP6 for wind/solar drought projections.
BENCHMARK: Partial — energy-drought classification from CMIP6 wind fields.

**[30] Baran & Lerch 2024 (2-step ML wind power) — QJRMS**
DOI: 10.1002/qj.4635
Two-stage ML: downscale CMIP wind → station-adaptive postprocessing for wind power.
BENCHMARK: Partial — station-adaptive postprocessing analogous to our approach.

---

## Recommended Benchmark Hierarchy

1. **GEV/return-value baseline** (Kharin 2013, Outten 2021) — fit GEV to annual wind maxima, compute exceedance probability
2. **ETCCDI percentile index** (Sillmann 2013) — standard climate extremes index methodology
3. **Logistic regression on climate indices** (Bochenek 2022) — simple ML lower bar
4. **DRN (Distributional Regression Network)** (Schulz & Lerch 2022) — leading ML method for wind extremes
5. **DeepESD CNN** (Baño-Medina 2022) — directly comparable CNN architecture on CMIP data

---

## BibTeX keys to add to ijcai23.bib

banomedinaDeepESD2022, zhangBiGRUWind2021, busterSRDRN2024, harderConstrainedDL2023,
srivastavaDownscaling2023, lianTerraWind2024, shenCMIP6Wind2022,
schulzLerchWindGusts2022, raspLerch2018, accarinоWGPNet2024,
outtenSobolowski2021, kumarCMIP5Wind2015, michelCORDEX2023,
reichsteinNature2019, hamENSO2019, kashinathClimateNet2021,
racahExtremeWeather2017, raspWeatherBench2020, molinaBenchmark2021, pathakFourCastNet2022,
sillmannCMIP5extremes2013, kharinCMIP5GEV2013, duebenBauer2018, schultzDLvsNWP2021,
bochenek2022, krishnanCMIP6wind2020, linTyphoon2025, sorianoCMIP6offshore2023,
zappaCORDEX2024, baranLerch2024
