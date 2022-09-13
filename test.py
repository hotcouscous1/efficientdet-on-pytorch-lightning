import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath('./src')))
from src.__init__ import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def test(cfg: DictConfig):
    from src.lightning_model import COCO_EfficientDet
    from src.dataset.val_dataset import Validate_Detection
    from torch.utils.data import DataLoader
    from src.utils.config_trainer import Config_Trainer

    # lightning model
    pl_model = COCO_EfficientDet(**cfg.model.model, **cfg.model.loss, **cfg.model.nms, **cfg.model.optimizer)

    # dataset and dataloader
    test_set = Validate_Detection(cfg.dataset.test.root, pl_model.model.img_size, cfg.dataset.dataset_stat)
    test_loader = DataLoader(test_set, batch_size=1)

    # trainer
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False, num_sanity_val_steps=0)

    # run test
    if 'ckpt_path' in cfg:
        trainer.test(pl_model, test_loader, ckpt_path=cfg.ckpt_path)
    else:
        raise RuntimeError('no checkpoint is given')


if __name__ == "__main__":
    test()
