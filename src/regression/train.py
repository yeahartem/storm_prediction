import sys,os
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
import torch

# PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
# pytorch_lightning 2.0.2 checkpoint loading (contains OmegaConf DictConfig).
# Monkey-patch to restore backward-compatible behavior.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import random
import logging
from datetime import datetime
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from src.regression.datamodule import WindDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.loggers import WandbLogger
# import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
# для mlflow dvc:
import shutil
import os
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import git


# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'out/{now:%Y-%m-%d}/{now:%H-%M-%S}')

@rank_zero_only
def log_config(cfg):
    params = OmegaConf.to_container(cfg, resolve=True)
    flat_params = flatten_dict(params)
    mlflow.log_params(flat_params)
# @rank_zero_only
# def log_config(cfg):
#     wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


@rank_zero_only
def log_model_arch(model):
    logging.info(model)

# --- ДОБАВЬ ЭТУ ВСПОМОГАТЕЛЬНУЮ ФУНКЦИЮ ---
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
# --- КОНЕЦ ВСПОМОГАТЕЛЬНОЙ ФУНКЦИИ ---

def train_regression(cfg: DictConfig) -> None:
    logging.info(f"Starting in {os.getcwd()}")
    start_time = time.process_time()
    # ------------------COMMENTED
    # os.environ['WANDB_API_KEY'] = '7ce4e8a3a21df6f25a3a589a9de3f52c759b3633'
    # os.environ['WANDB_MODE'] = 'offline'
    # # os.environ['WANDB_DIR'] = '/trinity/home/v.morozov/Wind/out/wandb'
    # # os.environ['WANDB_CONFIG_DIR'] = 'trinity/home/v.morozov/Wind/out/wandb'
    # # os.environ['WANDB_CACHE_DIR'] = 'trinity/home/v.morozov/Wind/out/wandb'
    # os.environ['WANDB_DIR'] = 'out/wandb'
    # os.environ['WANDB_CONFIG_DIR'] = 'out/wandb'
    # os.environ['WANDB_CACHE_DIR'] = 'out/wandb'
    # ------------------COMMENTED
    torch.set_float32_matmul_precision('high')
    run_dir = get_rundir_name()
    # wandb.init()
    # wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
    #                            project=cfg.project_name,
    #                            name=cfg.experiment_name,
    #                            log_model='all')
    mlflow_logger = MLFlowLogger(experiment_name=cfg.project_name,
                                 run_name=cfg.experiment_name,
                                 tracking_uri="file:./mlruns",
                                 log_model='all')
    # mlflow.log_artifact(os.path.join(os.getcwd(),"configs/cmip5_TestNet.yaml"), "config.yaml")
    # mlflow.log_artifact(os.path.join(os.getcwd(),"configs/cmip5_TestNet.yaml"), "config.yaml")
    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.log_param('git_commit_hash', repo.head.object.hexsha)
    except git.InvalidGitRepositoryError:
        logging.warning("Not a git repository. Cannot log commit hash.")

    dm = WindDataModule(cfg) # self.DPL = DataPreLoader(cfg)
    model = WindNetPL(cfg, run_dir)
    logging.info(f"Asking for {cfg.train.gpu_num} GPUs")
    logging.info(f"Visible is {torch.cuda.device_count()} GPUs")
    logging.info(f"torch version {torch.__version__ }")

    if torch.__version__ == "2.0.1" or torch.__version__ == "2.0.0" or  torch.__version__ == "2.0.1+cu117":
        # model.net = torch.compile(model.net)
        logging.info("Model compiled") # Почему коммент сверху?
    else:
        logging.info("PyTorch version is smaller than 2.0, compilation is not supported")

    default_root_dir = run_dir
    checkpoint_loc = run_dir 
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_loc, save_top_k=1, monitor="val/loss")
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=10,       # останавливаем если val/loss не улучшается 10 эпох подряд
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False)

    trainer = pl.Trainer(max_epochs=cfg.train.max_epoch,
                            profiler=None,
                            default_root_dir=default_root_dir,
                            callbacks=[lr_monitor, checkpoint_callback, early_stopping],
                            #performance
                            accelerator="gpu",
                            precision="16-mixed", # fp16 mixed precision: ~1.5-2x speedup on Ampere (A5000)
                            benchmark=True,
                            #validation
                            check_val_every_n_epoch=1,
                            num_sanity_val_steps=0, # Было 0 !!!
                            #distributed
                            devices=cfg.train.gpu_num,
                            num_nodes=cfg.train.num_nodes if cfg.train.distributed else 1,
                            strategy=cfg.train.strategy if cfg.train.distributed else 'auto',
                            #log
                            log_every_n_steps=cfg.train.log_every_n_steps,
                            limit_train_batches=cfg.train.get('limit_train_batches', 1.0),
                            limit_val_batches=cfg.train.get('limit_val_batches', 1.0),
                            # gradient_clip_val=1, # Чтобы не было ошибки inf в mlflow 
                            logger=mlflow_logger, # wandb_logger
                            #misc
                            # profiler='simple',
                            )
    log_config(cfg)

    # Сохраняем Git-хеш для воспроизводимости
    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.log_param('git_commit_hash', repo.head.object.hexsha)
    except git.InvalidGitRepositoryError:
        logging.warning("Not a git repository. Cannot log commit hash.")

    # Сохраняем финальный конфиг как артефакт
    # Hydra сохраняет его в папке .hydra в директории запуска
    final_config_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    if os.path.exists(final_config_path):
        mlflow.log_artifact(final_config_path, "config.yaml")

    # ---- NEW: test-only mode ----
    if getattr(cfg.train, "test_only", False):
        ckpt = getattr(cfg.train, "ckpt_path", None)
        if not ckpt:
            raise ValueError("cfg.train.test_only=True but cfg.train.ckpt_path is not set")
        logging.info(f"TEST-ONLY mode. Using checkpoint: {ckpt}")
        test_results = trainer.test(model=model, datamodule=dm, ckpt_path=ckpt)
        logging.info(f"Test results: {test_results}")
        return

    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)

    # ---- NEW: run TEST on best checkpoint ----
    logging.info("Running test on best checkpoint...")
    test_results = trainer.test(model=model, datamodule=dm, ckpt_path="best")
    logging.info(f"Test results: {test_results}")

    # (опционально) залогировать test метрики явно в mlflow
    if test_results and isinstance(test_results, list) and len(test_results) > 0:
        for k, v in test_results[0].items():
            try:
                mlflow.log_metric(f"test/{k}", float(v))
            except Exception:
                pass    
    # ======================== NEW CODE FOR SAVING CHECKPOINT FOR DVC ==================
    logging.info("Training finished. Copying best checkpoint for DVC...")

    # 1. Находим путь к лучшему чекпоинту, который сохранил Lightning
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        logging.info(f"Best checkpoint found at: {best_checkpoint_path}")
        mlflow.log_artifact(best_checkpoint_path, "model_checkpoints") # Сохраняем в mlflow
        # 2. Создаём папку назначения, если её нет
        destination_folder = 'model_weights'
        os.makedirs(destination_folder, exist_ok=True)
        destination_path = os.path.join(destination_folder, 'best_model.ckpt')

        # 3. Копируем файл
        shutil.copy(best_checkpoint_path, destination_path)
        logging.info(f"Checkpoint copied to: {destination_path}")
    else:
        logging.warning("Could not find best checkpoint path to copy.")


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip5_TestNet.yaml")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    train_regression(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    mlflow.end_run()
    # wandb.finish()
