import wandb
from pytorch_lightning.loggers.wandb import *
from pytorch_lightning.utilities.imports import _WANDB_GREATER_EQUAL_0_10_22
import shutil


class Another_WandbLogger(WandbLogger):

    __doc__ = r"""
        WandbLogger which is able to utilize wandb.Artifact

        Examples:
            logger = Another_WandbLogger(
                project='coco',
                config={'dataset': 'coco', 'model': 'effcientdet-d0'},
                artifact_name='d0',
                artifact_type='efficientdet',
                artifact_save_files={'trainer': 'config/trainer/default.yaml'},
            )
            logger.config.update({
                'epoch': 100,
                'batch_size': 64,
                'lr': 0.0001
            })

            artifact = logger.run.use_artifact('my-account/coco/d0:latest', type='efficientdet')
            artifact_dir = artifact.download()

            trainer = pytorch_lightning.Trainer(**cfg, logger=logger, log_every_n_steps=10)

        * If you want to give Another_WandbLogger with other loggers of pytorch_lightning, use 
        'trainer.loggers.append(Another_WandbLogger(...))'.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            project: Optional[str] = None,
            config: Union[Dict, str, None] = None,
            id: Optional[str] = None,
            save_dir: Optional[str] = None,
            save_every_epoch: Optional[bool] = True,
            offline: Optional[bool] = False,
            anonymous: Optional[bool] = None,
            artifact_name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            artifact_description: Optional[str] = None,
            artifact_save_files: Optional[Dict] = None,
            **kwargs
    ):

        log_model = 'all' if save_every_epoch else True

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        super().__init__(name=name, save_dir=save_dir, offline=offline, id=id, anonymous=anonymous,
                         project=project, log_model=log_model, **kwargs)

        self.log_hyperparams(config)
        self.config = self.experiment.config

        self._artifact_name = artifact_name if artifact_name else f"model-{self.experiment.id}"
        self._artifact_type = artifact_type if artifact_type else "model"
        self._artifact_description = artifact_description
        self._artifact_save_files = artifact_save_files

        self.run = self.experiment
        self.artifact = None


    def _scan_and_log_checkpoints(self, checkpoint_callback):
        # excepth the artifact, it is equal to the original
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }

        checkpoints = sorted((Path(p).stat().st_mtime, p, s) for p, s in checkpoints.items() if Path(p).is_file())
        checkpoints = [c for c in checkpoints if c[1] not in self._logged_model_time.keys()
                       or self._logged_model_time[c[1]] < c[0]]

        for t, p, s in checkpoints:
            metadata = (
                {
                    "score": s,
                    "original_filename": Path(p).name,
                    "ModelCheckpoint": {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        if hasattr(checkpoint_callback, k)
                    },
                }
                if _WANDB_GREATER_EQUAL_0_10_22
                else None
            )
            artifact = wandb.Artifact(name=self._artifact_name,
                                      type=self._artifact_type,
                                      description=self._artifact_description,
                                      metadata=metadata)

            artifact.add_file(p, name="model.ckpt")
            [artifact.add_file(f, name=n) for n, f in self._artifact_save_files.items()]

            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            self.artifact = artifact

            self._logged_model_time[p] = t


    @classmethod
    def remove_cache(cls, dir_path: str):
        if os.path.exists(dir_path):
            print("Every wandb cache files will be cleared. If you want, please enter Yes:")

            if input() in ['yes', 'Yes']:
                shutil.rmtree(dir_path)
            else:
                print('files are not cleared.')
        else:
            raise FileNotFoundError("no such directory exists")

