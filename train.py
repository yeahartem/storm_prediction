import sys,os
sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.regression.train import train_regression

    

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/train_configs"), config_name="conv_w_reg")
def main(cfg: DictConfig):    
    train_regression(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
