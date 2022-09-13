import os
import sys
import wandb
import argparse

sys.path.append(os.path.dirname(os.path.abspath('./src')))
from src.__init__ import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def train(cfg: DictConfig):
    from src.lightning_model import COCO_EfficientDet

    from src.dataset.train_dataset import COCO_Detection
    from src.dataset.val_dataset import Validate_Detection
    from src.dataset.bbox_augmentor import default_augmentor
    from src.dataset.utils import make_mini_batch
    from torch.utils.data import DataLoader

    from src.utils.config_trainer import Config_Trainer
    from src.utils.wandb_logger import Another_WandbLogger


    # lightning model
    pl_model = COCO_EfficientDet(**cfg.model.model, **cfg.model.loss, **cfg.model.nms, **cfg.model.optimizer,
                                 val_annFile=cfg.dataset.val.annFile)
    # augmentor
    augmentor = default_augmentor(pl_model.model.img_size)

    # dataset and dataloader
    train_set = COCO_Detection(cfg.dataset.train.root, cfg.dataset.train.annFile, augmentor)
    val_set = Validate_Detection(cfg.dataset.val.root, pl_model.model.img_size, cfg.dataset.dataset_stat)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              collate_fn=make_mini_batch, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers)

    # trainer
    logger = Another_WandbLogger(**cfg.log)

    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=logger, num_sanity_val_steps=0)

    logger.watch(pl_model)

    # run training
    if 'ckpt_path' in cfg:
        trainer.fit(pl_model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(pl_model, train_loader, val_loader)

    wandb.finish()


if __name__ == "__main__":
    train()

