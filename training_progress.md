
## 2026-04-10 13:45
- epoch=5-step=30000.ckpt
- epoch=6-step=35000.ckpt
- epoch_0_RMSE_vs_target.png
- epoch_0_predictions.csv
- epoch_1_RMSE_vs_target.png
- epoch_1_predictions.csv
- epoch_2_RMSE_vs_target.png
- epoch_2_predictions.csv
- epoch_3_RMSE_vs_target.png
- epoch_3_predictions.csv
- epoch_4_RMSE_vs_target.png
- epoch_4_predictions.csv
- epoch_5_RMSE_vs_target.png
- epoch_5_predictions.csv
- epoch_6_RMSE_vs_target.png
- epoch_6_predictions.csv

## 2026-04-10 13:45
- === val/AP ===
- === val/F1_score ===
- === val/precision ===
- === val/recall ===

## 2026-04-10 19:41
- 136:                print(f"--- Р’Р—Р’Р•РЁР•РќРќР«Р™ LOSS (РЁРђР“ {self.trainer.global_step}) ---")
- 140:                print(f"Р’Р·РІРµС€РµРЅРЅС‹Р№ Loss (РїРµСЂРІС‹Рµ 5):    {weighted_loss[:5].cpu().detach().numpy().round(2)}")
- 141:                print(f"Loss per sample ():     {per_sample_loss}")
- 155:            #     print(f"\n--- DEBUG: loss() step={self.trainer.global_step} ---")
- 156:            #     print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}")
- 160:            #     print(f"Loss per sample (first 5): {np.round(per_sample_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
- 162:            #     print(f"Weighted loss (first 5): {np.round(weighted_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
- 175:                print(f"\n--- DEBUG: loss() step={self.trainer.global_step} ---")
- 176:                print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}")
- 180:                print(f"Loss per sample ():     {per_sample_loss}")
- 342:        print(f"Р’ on_validation_epoch_end() -> preds.shape: {preds.shape}, target.shape: {targets.shape}")
- 345:        print("\n--- DEBUG: De-normalization on Epoch End ---")
- 571:        print(f"Р’ on_test_epoch_end() -> preds_float.shape: {preds_float.shape}, target_float.shape: {target_float.shape}")

## 2026-04-11 17:52
- 119:Accurate assessment of extreme wind risk is critical for infrastructure planning, disaster preparedness, and climate adaptation, yet remains fundamentally limited by the sparse and non-uniform distribution of surface weather stations. Climate models provide global spatial coverage but systematically underestimate local wind extremes due to spatial averaging. We propose a deep learning framework that bridges this gap by training a convolutional neural network to predict the probability of a locally extreme wind event at any location on a global grid, using gridded CMIP6 climate fields as input and ground-truth labels derived from over \textcolor{brown}{XXX} GSOD weather stations worldwide. The target is defined as a station-adaptive binary indicator: whether the maximum observed wind speed within a 28-day window exceeds a location-specific threshold calibrated to the 95th percentile of local wind climatology. Evaluated on a held-out test period (2023--2024), our model achieves an AUROC of \textcolor{brown}{XXX} globally, with consistent performance across climatically diverse regions including Europe, Siberia, and sub-Saharan Africa. The trained model can be applied directly to CMIP6 future scenario projections, enabling global storm-risk maps at $1.125^\circ$ resolution up to the year 2100.
- 131:    \includegraphics[width=\linewidth]{pics/stations_map.png}
- 132:    \caption{Global distribution of GSOD weather stations used in this study. Station density is highest over North America and Europe and sparse over central Africa, Siberia, and oceanic regions---a pattern representative of the data scarcity our method is designed to address.}
- 136:Climate models offer global spatial coverage but introduce a different limitation: by construction they represent grid-cell averages, systematically smoothing out the local extremes that drive damage. The sixth phase of the Coupled Model Intercomparison Project (CMIP6) provides daily fields on a $\sim$1$^\circ$ grid, capturing large-scale atmospheric dynamics but attenuating point-scale wind peaks by a factor of two or more relative to station observations~\cite{SILLMANN201765}.
- 138:We propose to bridge this gap with a data-driven calibration: a deep convolutional neural network (CNN) trained to predict, for each grid cell, the probability that the local wind speed will exceed a station-specific extreme threshold within a 28-day window, conditioned on the surrounding CMIP6 climate state. Once trained on historical data (2000--2022), the model can be driven by any CMIP6 scenario projection to produce global storm-risk maps up to 2100.
- 142:    \item We introduce a station-adaptive target definition that combines a 28-day temporal aggregation window with a per-station 95th-percentile threshold, making the binary label both physically interpretable and robust across climatically heterogeneous regions (Section~\ref{sec:methodology}).
- 144:    \item We conduct an ablation study showing the marginal contribution of topographic elevation as an additional input channel, and compare against two interpretable baselines derived from raw CMIP6 fields (Section~\ref{sec:experiments}).
- 160:Statistical downscaling trains a statistical or ML model to map coarse climate model output to fine-scale observations, directly analogous to our task of mapping CMIP6 grid cells to station-level extreme probabilities. \cite{banomedinaDeepESD2022} introduced DeepESD, a CNN-based perfect-prognosis downscaler for the CMIP5 multi-model ensemble, demonstrating that spatial convolutional architectures substantially outperform traditional regression-based methods. \cite{busterSRDRN2024} extend this with a super-resolution deep residual network combined with quantile delta mapping to simultaneously bias-correct and downscale six CMIP6 variables including surface wind over CONUS, addressing non-stationarity critical for future projections. \cite{harderConstrainedDL2023} introduce hard physical constraints (conservation-law renormalization layers) into CNN downscaling, improving out-of-sample accuracy. \cite{shenCMIP6Wind2022} quantify global CMIP6 systematic wind biases, showing that raw CMIP6 magnitudes systematically underestimate local extremes -- the core motivation for a learned calibration layer. \cite{gonzalez2020calibration} calibrate ERA5 wind using building-mounted sensors via quantile matching and random forests, showing that local corrections substantially improve agreement with point observations.
- 163:\cite{schulzLerchWindGusts2022} conduct the most comprehensive comparison to date of ML methods for probabilistic wind gust forecasting at surface stations, including distributional regression networks (DRN), gradient-boosted EMOS, and quantile regression forests, on 175 German stations using a convection-permitting ensemble (COSMO-D2-EPS); the DRN achieves state-of-the-art skill. \cite{raspLerch2018} demonstrated that neural networks with station-specific embeddings outperform parametric EMOS for ensemble postprocessing, a design principle directly relevant to our station-adaptive threshold. \cite{lledo2020predicting} combine the S2S sub-seasonal forecast database~\cite{vitart2017subseasonal} with the Madden--Julian Oscillation phase to improve medium-range wind forecasts over Europe--Atlantic. These approaches post-process existing forecasts at individual stations or on a fixed grid, but do not produce spatially complete risk maps applicable to unobserved locations under future climate projections.
- 166:Several recent studies use CMIP6 for downstream wind impact modelling. \cite{zhangBiGRUWind2021} train a BiGRU network to downscale CMIP6 near-surface wind speed to 0.25$^\circ$ for offshore wind resource projections in China to 2100. \cite{sorianoCMIP6offshore2023} combine bias-corrected CMIP6 wind projections with percentile-based classification of offshore wind energy adequacy along the Spanish coast -- one of the few papers applying a classification (rather than regression) framework to CMIP6 wind, making it a direct methodological comparator. \cite{krishnanCMIP6wind2020} show that CMIP6 improves over CMIP5 for Bay of Bengal near-surface wind but systematic biases remain, justifying our station-adaptive threshold design. \cite{zappaCORDEX2024} train an ML emulator on CMIP5-CORDEX simulations and transfer it to CMIP6 for wind energy drought projections, demonstrating that ML-based climate downscaling can bridge CMIP generations.
- 168:In summary, the existing literature addresses the component pieces of our problem -- GEV-based extreme analysis, CNN classification of extreme events from gridded data, statistical downscaling of climate model output -- but no prior work combines all three: a global, station-validated, binary extreme wind classifier trained end-to-end on CMIP6 fields with a station-adaptive target and evaluated on a held-out future period.
- 210:\textbf{Station observations.} For ground-truth labels we use the Global Surface Summary of the Day (GSOD) dataset~\cite{gsod23}, which archives daily observations from over 9{,}000 weather stations worldwide covering the period 1929--present. After quality filtering (stations with fewer than 25 valid observations in any 6-month window are discarded), our global training set retains approximately \textcolor{brown}{XXX} stations distributed across all continents. The non-uniform spatial distribution of this observational network---densest over North America and Europe, sparser over Africa and Central Asia---is illustrated in Figure~\ref{fig:stations_coverage}.
- 241:         \includegraphics[height=3.3cm, width=\linewidth]{pics/stations_temperature2.png}
- 247:         \includegraphics[height=3.3cm, width=\linewidth]{pics/stations_wind2.png}
- 251:        \caption{Violins plots of temperatures and wind speeds from climate model and the ones measured on weather stations}
- 255:Figure~\ref{fig:violins_data} compares the distributions of temperature (a,~c) and daily maximum wind speed (b,~d) from the CMIP6 MRI-ESM2-0 model and GSOD station measurements over the global training period (2000--2022). The plots show strong agreement in the median, interquartile range, and overall shape for both variables. This concordance confirms that CMIP6 provides a suitable physical basis for training the predictive model, even though the model's grid-cell averages systematically attenuate the tails of the wind speed distribution relative to point observations---the core challenge our method addresses.
- 347:If a pixel contains more than one weather station, the maximum wind speed across all co-located stations is used when evaluating the condition in Eq.~\ref{eq:target}. With these definitions, the fraction of positive labels in the global dataset is approximately 42\%, reflecting the combination of the 28-day aggregation window and the station-adaptive threshold.
- 419:where $N_{\mathcal{D}} = |\mathcal{S}_{\mathcal{D}}| \times T$ with $|\mathcal{S}_{\mathcal{D}}|$ the number of station-occupied pixels and $T$ the number of valid time steps after alignment. Each input patch has shape $C \times \Delta_t \times (2w{+}1) \times (2w{+}1)$, with $C=5$ channels (four CMIP6 variables plus static elevation), temporal window $\Delta_t = 27$ days, and spatial half-width $w = 47$ pixels ($95 \times 95$ grid cells, covering roughly $106^\circ \times 106^\circ$ at the equator).
- 442:All experiments are conducted on a server equipped with two NVIDIA RTX A5000 GPUs (24~GB VRAM each), 256~GB RAM, and 32 CPU cores. Training uses PyTorch~\cite{paszke2019pytorch} with the PyTorch Lightning framework and \texttt{fp16} mixed precision. Experiments are tracked with MLflow; model checkpoints are saved after every epoch and the best checkpoint by validation loss is used for final evaluation. The code and configuration files are available at \textcolor{brown}{[repository URL]}.
- 453:         Method & AUROC & AP & F1 \\
- 573:We presented a deep learning framework for global extreme wind risk assessment that bridges the resolution gap between coarse climate model output and point-scale station observations. A convolutional neural network is trained to predict, for every CMIP6 grid cell, the probability that a locally extreme wind event will occur within a 28-day window, using a station-adaptive threshold that combines a location-specific 95th-percentile criterion with a 15~m/s physical lower bound. Trained on over \textcolor{brown}{XXX} GSOD stations worldwide (2000--2022) and evaluated on a held-out 2023--2024 test period, the model achieves a global AUROC of \textcolor{brown}{XXX}, substantially outperforming both the raw-CMIP threshold baseline (F1\,=\,0.19) and logistic regression (F1\,=\,0.19). Performance is consistent across climatically diverse regions including Russia, sub-Saharan Africa, and South and East Asia, suggesting that the learned spatial features generalise well beyond the station-rich areas that dominate the training signal.

## 2026-04-11 17:55
- 442:All experiments are conducted on a server equipped with two NVIDIA RTX A5000 GPUs (24~GB VRAM each), 256~GB RAM, and 32 CPU cores. Training uses PyTorch~\cite{paszke2019pytorch} with the PyTorch Lightning framework and \texttt{fp16} mixed precision. Experiments are tracked with MLflow; model checkpoints are saved after every epoch and the best checkpoint by validation loss is used for final evaluation. The code and configuration files are available at \textcolor{brown}{[repository URL]}.

## 2026-04-11 19:29
- # import shap
- sys.path.append('..')
- from src.data_assemble.wrap_data import *
- from src.data_assemble.wrap_data import *
- trainer = pl.Trainer(max_epochs=50,
- check_val_every_n_epoch=1,
- # chk_path = "./lightning_logs/version_13/checkpoints/epoch=35-step=288.ckpt"
- # Shap
- # e = shap.DeepExplainer(model2, background)
- # shap_values = e.shap_values(test_images)
- #     print(features[i], abs((torch.mean(torch.tensor(shap_values[0]), dim=[0, 2, 3])[i]).numpy()) + (torch.mean(torch.tensor(shap_values[1]), dim=[0, 2, 3])[i]).numpy() )
- # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
- # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
- # shap.image_plot(shap_numpy, -test_numpy)
- grid = np.zeros((grid_inference[some_key].shape[0], max_x, max_y))
- for i in range(grid.shape[0]):
- for t in list(range(grid_new.shape[0]))[::100]:
- # ax.set_xticks(np.arange(0,grid_new.shape[2],2), labels=X)
- # ax.set_yticks(np.arange(grid_new.shape[1]-1,-1,-2), labels=Y)
- # filenames = [os.path.join('tmp_dump', str(f) + '.png') for f in list(range(grid.shape[0]))[::100]]
- #     images.append(imageio.imread(filename))
- # imageio.mimsave('tmp_dump/wind_prob_maps.gif', images, duration=0.5)
- # Map visualisation
- Y_full.append(float(coord[k][0].data))
- X_full.append(float(coord[k][1].data))
- dff = map_to_pandas(grid=grid, x_axis=X_full, y_axis=Y_full, t_axis=time_axis, start_date='', day_interval=day_interval)
- plot_map(df[df['date']==dates[51]], 'value', part_world_to_plot='Florida', vmin=0, vmax=1, text = '21-09-2022')
- for t in list(range(grid.shape[0]))[::day_interval]:
- plot_map(df[df['date'] == date], column, part_world_to_plot='KK', vmin=vmin, vmax=vmax,
- # Picture for Paper

## 2026-04-11 19:43
- epoch=5-step=30000.ckpt
- epoch=6-step=35000.ckpt
- epoch_0_RMSE_vs_target.png
- epoch_0_predictions.csv
- epoch_1_RMSE_vs_target.png
- epoch_1_predictions.csv
- epoch_2_RMSE_vs_target.png
- epoch_2_predictions.csv
- epoch_3_RMSE_vs_target.png
- epoch_3_predictions.csv
- epoch_4_RMSE_vs_target.png
- epoch_4_predictions.csv
- epoch_5_RMSE_vs_target.png
- epoch_5_predictions.csv
- epoch_6_RMSE_vs_target.png
- epoch_6_predictions.csv
- epoch=11-step=60000.ckpt
- epoch=6-step=35000.ckpt
- epoch_0_RMSE_vs_target.png
- epoch_0_predictions.csv
- epoch_10_RMSE_vs_target.png
- epoch_10_predictions.csv
- epoch_11_RMSE_vs_target.png
- epoch_11_predictions.csv
- epoch_12_RMSE_vs_target.png
- epoch_12_predictions.csv
- epoch_13_RMSE_vs_target.png
- epoch_13_predictions.csv
- epoch_1_RMSE_vs_target.png
- epoch_1_predictions.csv

## 2026-04-11 19:43
- keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
- cfg: {'raw': {'paths_to_climate_files_folders': ['./data/cmip6_world_orig'], 'path_to_elevation': 'data/elevation.nc', 'path_to_weather_stations_data': 'data/weatherstation_data/data_meteo_full.parquet', 'path_to_weather_station_list': 'data/weatherstation_data/weatherstation_list.json', 'path_to_world_weather_stations_data': 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'}, 'process': {'data_dir': 'data/cmip6_world/', 'prepared_target_data_name': 'target.parquet', 'make_climate_data': True, 'make_elevation_data': False, 'make_normalization': True, 'make_cleaned_weather_data': True, 'make_target': True, 'load_normalization': False, 'saved_normalized': True, 'precision': 16, 'spatial_crop': False, 'coords': {'lat_min': -47.0, 'lat_max': 80.0, 'lon_min': -180.0, 'lon_max': 180.0}, 'min_height': -10, 'max_height': 800, 'variables': ['sfcWindmax', 'pr', 'tasmax', 'tasmin'], 'time_limits': ['2000-01-01', '2024-12-31'], 'start_of_test': '2023-01-01', 'target_column': ['max_speed']}, 'train': {'make_tmp_target_file': False, 'use_elevation': False, 'normalize': False, 'time_agg_window': 28, 'time_freq': 14, 'target_threshold': 10, 'data_dir': 'data/cmip6_world/', 'target_data_file': 'target.parquet', 'start_time': '2000-01-01', 'end_time': '2024-12-31', 'start_of_test': '2023-01-01', 'variables': ['sfcWindmax', 'pr', 'tasmax', 'tasmin'], 'spatial_crop': False, 'lat_min': 30, 'lat_max': 50, 'lon_min': -30, 'lon_max': 40, 'batch_size': 16, 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'max_epoch': 100, 'optimizer_name': 'AdamW', 'scheduler_name': 'LinearLR', 'loss_name': 'BCELoss', 'num_workers': 0, 'gpu_num': 1, 'distributed': False, 'num_nodes': 1, 'strategy': 'ddp', 'log_every_n_steps': 10, 'test_only': False, 'ckpt_path': 'mlruns/680925330068273108/0307a5d955cf4cabbfce7175f20e3d5b/artifacts/model/checkpoints/epoch=1-step=1000/epoch=1-step=1000.ckpt', 'cache': {'use_cached_padding': False, 'pad_crop_data_path': 'data/cache/cmip6_pad_crop_data.npy', 'shift_data_path': 'data/cache/cmip6_shift_data.pkl'}}, 'eval': {'wind_risk_threshold': 20, 'use_elevation': False, 'time_start': '2018-06-06', 'time_end': '2018-12-12', 'lat_min': 28.5, 'lat_max': 68.5, 'lon_min': -46.5, 'lon_max': 78.5, 'interpolation_res': 0.2, 'num_workers_eval': 16, 'batch_size_test': 60, 'distributed_test': False, 'strategy': 'ddp', 'gpu_num': 1, 'save_kml': True, 'data_dir': 'data/cmip6_1deg_world/', 'target_data_file': 'target.parquet', 'path_to_checkpoint': '/app/wind/out/wind.ckpt', 'path_to_predictions': 'out/predictions/result_world_cmip6.csv', 'output_file': 'out/predictions/output_file_world_cmip6', 'cmip_type': 'CMIP6'}, 'experiment_name': 'cmip6_world_GhostWindNet27x47_BCE', 'project_name': 'wind-speed-classification-cmip6', 'target_type': 'wind_ms', 'cmip_type': 'CMIP6', 'model_name': 'GhostWindNet27', 'time_window': 27, 'half_side_size': 47, 'start_date': '2023-01-01', 'end_date': '2024-12-31'}

## 2026-04-12 11:16
- 2026-04-12 11:16:23,126 INFO Elevation padded shape: (254, 414)
- 2026-04-12 11:16:23,137 INFO Loading checkpoint: out/2026-04-03/12-19-06/epoch=11-step=60000.ckpt
- 2026-04-12 11:16:23,526 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:16:29,024 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.485, 0.780]
- 2026-04-12 11:16:29,069 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:16:34,105 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.490, 0.820]
- 2026-04-12 11:16:34,147 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:16:39,186 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.509, 0.843]
- 2026-04-12 11:16:39,202 WARNING cartopy not found, plotting without map background

## 2026-04-12 11:18
- 2026-04-12 11:18:14,580 INFO Elevation padded shape: (254, 414)
- 2026-04-12 11:18:14,590 INFO Loading checkpoint: out/2026-04-03/12-19-06/epoch=11-step=60000.ckpt
- 2026-04-12 11:18:14,928 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:18:19,827 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.485, 0.780]
- 2026-04-12 11:18:19,864 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:18:24,480 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.490, 0.820]
- 2026-04-12 11:18:24,517 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:18:29,113 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.509, 0.843]
- 2026-04-12 11:18:29,118 WARNING cartopy unavailable (DLL load failed while importing _network: пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅ пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ %1.), plotting without map background

## 2026-04-12 11:19
- 2026-04-12 11:19:36,452 INFO Elevation padded shape: (254, 414)
- 2026-04-12 11:19:36,462 INFO Loading checkpoint: out/2026-04-03/12-19-06/epoch=11-step=60000.ckpt
- 2026-04-12 11:19:36,822 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:19:41,914 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.485, 0.780]
- 2026-04-12 11:19:41,953 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:19:46,770 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.490, 0.820]
- 2026-04-12 11:19:46,807 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:19:51,527 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.509, 0.843]
- 2026-04-12 11:19:51,552 WARNING Could not load border shapefiles (DLL load failed while importing _network: пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅ пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ %1.)

## 2026-04-12 11:20
- 2026-04-12 11:20:41,535 INFO Elevation padded shape: (254, 414)
- 2026-04-12 11:20:41,546 INFO Loading checkpoint: out/2026-04-03/12-19-06/epoch=11-step=60000.ckpt
- 2026-04-12 11:20:41,895 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:20:46,885 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.485, 0.780]
- 2026-04-12 11:20:46,922 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:20:51,637 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.490, 0.820]
- 2026-04-12 11:20:51,674 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:20:56,388 INFO Grid shape: (22, 35), cells with data: 770, prob range: [0.509, 0.843]
- 2026-04-12 11:20:56,392 WARNING Could not resolve shapefile paths (DLL load failed while importing _network: пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅ пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ %1.)

## 2026-04-12 11:21
- 2026-04-12 11:21:39,069 INFO Elevation padded shape: (254, 414)
- 2026-04-12 11:21:39,078 INFO Loading checkpoint: out/2026-04-03/12-19-06/epoch=11-step=60000.ckpt
- 2026-04-12 11:21:39,428 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:21:44,351 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:21:48,993 INFO Combined padded shape: (4, 27, 254, 414)
- 2026-04-12 11:21:53,587 INFO Shapefiles: coast=ne_50m_coastline.shp, countries=ne_110m_admin_0_countries.shp, states=none

## 2026-04-12 12:37
- 1568052 Add notemp ablation config, inference_map.py, update Table 2 with nopr results
- 01972df Add EarlyStopping(patience=10, monitor=val/loss)
- 7d518c5 Speed: fp16-mixed precision, limit_train_batches=5000, remove gradient debug
- ac19ce9 Fix syntax: escaped newline in seasonal metrics print
- 61ee3b9 Set test_only mode for epoch=18 checkpoint evaluation
- ba07e9a CMIP6 global pipeline: BCELoss, GhostWindNet27, data bug fixes

## 2026-04-12 22:39
- 53:            if cfg.train.loss_name=='MSELoss':
- 54:                self.criterion = torch.nn.MSELoss()
- 55:            elif cfg.train.loss_name=='L1Loss':
- 56:                self.criterion = torch.nn.L1Loss()
- 57:            elif cfg.train.loss_name=='MSELoss_Dense':
- 58:                self.criterion = torch.nn.MSELoss(reduction='none')
- 59:            elif cfg.train.loss_name=='L1Loss_Dense':
- 60:                self.criterion = torch.nn.L1Loss(reduction='none')
- 61:            elif cfg.train.loss_name=='BCELoss':
- 63:                self.criterion = nn.BCEWithLogitsLoss()
- 65:                raise NotImplementedError(f'Criterion {cfg.train.loss_name} not found')
- 67:        self.train_loss = MeanMetric()
- 68:        self.val_loss = MeanMetric()
- 69:        self.test_loss = MeanMetric()
- 103:        """Classification score: sigmoid for BCELoss (logits), float_to_score for regression."""
- 104:        if self.cfg.train.loss_name == 'BCELoss':
- 109:        """Binary prediction: logit > 0 for BCELoss, float_to_binary for regression."""
- 110:        if self.cfg.train.loss_name == 'BCELoss':
- 114:    def loss(self, y_hat, y, dense_weights):
- 116:        if self.cfg.train.loss_name=='MSELoss_Dense' or self.cfg.train.loss_name=='L1Loss_Dense':
- 126:            per_sample_loss = self.criterion(y_hat_squeezed, y)
- 129:            weighted_loss = per_sample_loss * dense_weights_squeezed
- 134:            return torch.mean(weighted_loss)
- 137:            # per_sample_loss = self.criterion(y_hat.squeeze(), y)
- 140:            # weighted_loss = per_sample_loss * dense_weights
- 144:            #     print(f"\n--- DEBUG: loss() step={self.trainer.global_step} ---")
- 149:            #     print(f"Loss per sample (first 5): {np.round(per_sample_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
- 151:            #     print(f"Weighted loss (first 5): {np.round(weighted_loss.flatten()[:5].cpu().detach().numpy(), 2)}")
- 156:            # return torch.mean(weighted_loss)
- 157:        elif self.cfg.train.loss_name=='BCELoss':

## 2026-04-13 11:28
- [experiment/no-elevation b4f07e4] Switch early stopping and checkpoint monitor from val/loss to val/AP

## 2026-04-13 — Test results: full model (cmip6_world_elevation_server)
Checkpoint: out/2026-04-12/21-12-58/epoch=12-step=65000.ckpt
Config: undersampling 1:1, label_smoothing=0.05, weight_decay=0.001, gradient_clip=1.0, time_freq=14, 1 station/cell

### Global test metrics
| Metric        | Value  |
|---------------|--------|
| test/AUROC    | 0.8470 |
| test/AP       | 0.7479 |
| test/F1       | —      |
| test/precision| 0.6538 |
| test/recall   | 0.7604 |
| Brier Score   | 0.1564 |
| BSS           | 0.3184 |
| clim_rate     | 0.357  |

### Seasonal metrics
| Season | AUROC  | AP     | F1     | pos_rate |
|--------|--------|--------|--------|----------|
| DJF    | 0.8706 | 0.7926 | 0.7556 | 0.394    |
| MAM    | 0.8589 | 0.7857 | 0.7198 | 0.369    |
| JJA    | 0.8171 | 0.6721 | 0.6320 | 0.311    |
| SON    | 0.8364 | 0.7151 | 0.6903 | 0.356    |

### Regional metrics
| Region              | AUROC  | AP     | F1     | pos_rate | n      |
|---------------------|--------|--------|--------|----------|--------|
| russia_europe       | 0.8491 | 0.6597 | 0.6223 | 0.243    | 9315   |
| russia_west_siberia | 0.8647 | 0.6288 | 0.6036 | 0.185    | 3476   |
| russia_east_siberia | 0.8069 | 0.3418 | 0.3565 | 0.046    | 2855   |
| russia_far_east     | 0.8466 | 0.6284 | 0.5893 | 0.217    | 2283   |
| africa_equatorial   | 0.8384 | 0.4896 | 0.5490 | 0.149    | 496    |
| africa_south        | 0.7564 | 0.6726 | 0.6046 | 0.330    | 2266   |
| africa_north_east   | 0.8134 | 0.6240 | 0.6419 | 0.301    | 1779   |
| africa_sahel_east   | 0.7444 | 0.4650 | 0.3850 | 0.178    | 594    |

### Confusion matrix (test set)
TN=63727  FP=18352  FN=10931  TP=34606

## 2026-04-13 23:54
- [experiment/no-elevation c04e1f7] Add per-epoch resampling, CosineAnnealingLR, psl variable to all server configs

## 2026-04-14 00:03
- [experiment/no-elevation c97b63a] Fix CosineAnnealingLR T_max: use cosine_t_max=20 instead of max_epoch=100
