import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.__init__ import *
from src.utils.bbox import convert_bbox, untransform_bbox
from src.dataset.metric import Evaluate_COCO
from src.model.efficientdet import EfficientDet
from src.loss.focal_loss import Focal_Loss
from src.utils.nms.hard_nms import Hard_NMS



class COCO_EfficientDet(pl.LightningModule):

    __doc__ = r"""
        This class is to manage the hyper-parameters at once, involved in training.
    
        Args:
            coeff: coefficient for EfficientDet
            pretrained_backbone: load checkpoints to the model's backbone
            ckpt_path: checkpoint path of only EfficientDet, not COCO_EfficientDet
                if you load the checkpoints of COCO_EfficientDet, use 'load_from_checkpoint'
            fore_th: foreground threshold for the loss function
            back_th: background threshold for the loss function
            alpha: alpha for focal-loss
            gamma: gamma for focal-loss
            beta: beta for smooth-L1 loss
            fore_mean: average the loss values by the number of foregrounds
            reg_weight: weight for smooth-L1 loss 
            average: average the loss values by the number of mini-batch
            iou_th: IoU threshold for Soft-NMS
            conf_th: confidence or score threshold for Soft-NMS
            gaussian: gaussian penalty for Soft-NMS
            sigma: sigma for Soft-NMS
            max_det: max detection number after Soft-NMS
            lr: learning rate  
            lr_exp_base: gamma for the exponential scheduler
            warmup_epochs: warm-up start epochs for the exponential scheduler
            val_annFile: file path of annotation for validation(instances_train2017.json) 
        
        * You can alter NMS or optimizer by modifications of lines.
    """

    def __init__(self,
                 coeff: int,
                 pretrained_backbone: bool = True,
                 ckpt_path: str = None,
                 fore_th: float = 0.5,
                 back_th: float = 0.4,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 beta: float = 0.1,
                 fore_mean: bool = True,
                 reg_weight: float = None,
                 average: bool = True,
                 iou_th: float = 0.5,
                 max_det: Optional[int] = 400,
                 lr: float = 1e-4,
                 val_annFile: str = None,
                 ):

        super().__init__()
        self.save_hyperparameters()

        self.coeff = coeff
        self.pretrained_backbone = pretrained_backbone
        self.ckpt_path = ckpt_path
        self.fore_th = fore_th
        self.back_th = back_th
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.fore_mean = fore_mean
        self.reg_weight = reg_weight
        self.average = average
        self.iou_th = iou_th
        self.max_det = max_det
        self.lr = lr
        self.annFile = val_annFile

        self.model = self.configure_model()
        self.anchors = self.model.anchors
        self.loss = self.configure_loss_function()
        self.nms = self.configure_nms()

        self.val_result_dir = None
        self.test_result_dir = None


    def configure_model(self):
        model = EfficientDet(self.coeff, 80, False, self.pretrained_backbone)

        if not self.pretrained_backbone:
            self.initialize_weight(model)
        else:
            self.initialize_weight(model.fpn)
            self.initialize_weight(model.head)

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path)
            assert isinstance(ckpt, OrderedDict), 'please load EfficientDet checkpoints'
            assert next(iter(ckpt)).split('.')[0] != 'model', 'please load EfficientDet checkpoints'
            model.load_state_dict(torch.load(self.ckpt_path))

        return model


    def configure_loss_function(self):
        return Focal_Loss(self.fore_th, self.back_th, self.alpha, self.gamma, self.beta,
                          self.fore_mean, self.reg_weight, self.average, 'cxcywh')

    def configure_nms(self):
        return Hard_NMS(self.iou_th, self.max_det, 'cxcywh')


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                      threshold=0.001, threshold_mode='abs', verbose=True)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": 'AP'}}


    @classmethod
    def initialize_weight(cls, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, input, detect):
        return self.model(input, detect)


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds, anchors = self.model(inputs, detect=False)
        sync_labels = convert_bbox(labels, 'xywh', 'cxcywh')

        loss, cls_loss, reg_loss = self.loss(preds, anchors, sync_labels)
        self.log('train_loss', loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_reg_loss', reg_loss)

        return loss


    def validation_step(self, batch, batch_idx):
        ids, inputs, scales, pads = batch
        preds, _ = self.model(inputs, detect=True)
        preds = self.nms(preds)

        for i, (scale, pad) in enumerate(zip(scales, pads)):
            preds[i] = convert_bbox(preds[i], 'cxcywh', 'xywh')
            preds[i] = untransform_bbox(preds[i], scale, pad, 'xywh')

        return ids, preds


    def test_step(self, batch, batch_idx):
        ids, inputs, scales, pads = batch
        preds, _ = self.model(inputs, detect=True)
        preds = self.nms(preds)

        for i, (scale, pad) in enumerate(zip(scales, pads)):
            preds[i] = convert_bbox(preds[i], 'cxcywh', 'xywh')
            preds[i] = untransform_bbox(preds[i], scale, pad, 'xywh')

        return ids, preds


    def validation_epoch_end(self, val_step):
        result = OrderedDict()

        for batch in val_step:
            for id, pred in zip(*batch):
                result[id] = pred

        if not self.val_result_dir:
            self.val_result_dir = os.path.join('result/val', datetime.datetime.now().strftime("run-%Y-%m-%d-%H-%M"))
        result_file = os.path.join(self.val_result_dir, 'epoch-{}.json'.format(self.current_epoch))

        val_metric = Evaluate_COCO(result, result_file, self.annFile, test=False)

        self.log('AP', val_metric['AP'])
        self.log('AP50', val_metric['AP50'])
        self.log('AR', val_metric['AR'])


    def test_epoch_end(self, test_step):
        result = OrderedDict()

        for batch in test_step:
            for img_id, pred in zip(*batch):
                result[img_id] = pred

        if not self.test_result_dir:
            self.test_result_dir = 'result/test'
        result_file = os.path.join(self.test_result_dir, datetime.datetime.now().strftime("run-%Y-%m-%d-%H-%M.json"))

        Evaluate_COCO(result, result_file, None, test=True)

