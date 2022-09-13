from src.__init__ import *
from src.model.anchor import Anchor_Assigner


class Focal_Loss(nn.Module):

    def __init__(self,
                 fore_th: float,
                 back_th: float,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 beta: float = 0.1,
                 fore_mean: bool = True,
                 reg_weight: Optional[float] = None,
                 average: bool = True,
                 bbox_format: str = 'cxcywh'
                 ):

        super().__init__()

        self.fore_th = fore_th
        self.back_th = back_th
        self.anchor_assigner = Anchor_Assigner(fore_th, back_th, False, False, bbox_format)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.fore_mean = fore_mean

        self.reg_weight = reg_weight if reg_weight else 1.0
        self.average = average


    @classmethod
    def focal_loss(cls, cls_pred, fore_idx, back_idx, fore_label_cls, alpha, gamma, mean):

        fore_pred = cls_pred[fore_idx]
        back_pred = cls_pred[back_idx]

        fore_pred_t = torch.where(fore_label_cls == 1, fore_pred, 1 - fore_pred)
        back_pred_t = 1 - back_pred

        fore_alpha_t = torch.where(fore_label_cls == 1, alpha, 1 - alpha)
        back_alpha_t = 1 - alpha

        fore_weight = -1 * fore_alpha_t * torch.pow(1 - fore_pred_t, gamma)
        back_weight = -1 * back_alpha_t * torch.pow(1 - back_pred_t, gamma)

        fore_loss = fore_weight * torch.log(fore_pred_t)
        back_loss = back_weight * torch.log(back_pred_t)

        loss = torch.sum(fore_loss) + torch.sum(back_loss)
        if mean:
            num = fore_idx.size(0)
            loss = loss / num if num > 0 else loss

        return loss


    @classmethod
    def smooothL1_loss(cls, reg_pred, anchors, fore_idx, fore_label_bbox, beta, mean):
        fore_pred = reg_pred[fore_idx]
        fore_anchor = anchors.squeeze()[fore_idx]

        reg_label = torch.zeros_like(fore_label_bbox)

        reg_label[..., 0] = (fore_label_bbox[..., 0] - fore_anchor[..., 0]) / fore_anchor[..., 2]
        reg_label[..., 1] = (fore_label_bbox[..., 1] - fore_anchor[..., 1]) / fore_anchor[..., 3]
        reg_label[..., 2] = torch.log(fore_label_bbox[..., 2].clamp(min=1) / fore_anchor[..., 2])
        reg_label[..., 3] = torch.log(fore_label_bbox[..., 3].clamp(min=1) / fore_anchor[..., 3])

        mae = torch.abs(reg_label - fore_pred)

        loss = torch.where(torch.le(mae, beta), 0.5 * (mae ** 2) / beta, mae - 0.5 * beta)
        loss = torch.sum(loss)
        if mean:
            num = 4 * fore_idx.size(0)
            loss = loss / num if num > 0 else loss

        return loss


    def forward(self,
                preds: Tensor,
                anchors: Tensor,
                labels: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:

        if len(preds.shape) != 3:
            raise ValueError("preds should be given in 3d tensor")

        if len(anchors.shape) != 3:
            raise ValueError("anchors should be given in 3d tensor")

        if len(labels.shape) != 3:
            raise ValueError("labels should be given in 3d tensor")

        reg_preds = preds[..., :4]
        cls_preds = preds[..., 4:]
        cls_preds = cls_preds.clamp(1e-5, 1.0 - 1e-5)

        target_assigns = self.anchor_assigner(labels, anchors)
        cls_losses, reg_losses = [], []

        for i, assign in enumerate(target_assigns):
            fore_idx = assign['foreground'][0]
            back_idx = assign['background'][0]

            fore_label_cls = assign['foreground'][1][..., 4:]
            fore_label_bbox = assign['foreground'][1][..., :4]

            cls_losses.append(self.focal_loss(cls_preds[i], fore_idx, back_idx, fore_label_cls, self.alpha, self.gamma, self.fore_mean))
            reg_losses.append(self.smooothL1_loss(reg_preds[i], anchors, fore_idx, fore_label_bbox, self.beta, self.fore_mean))

        cls_loss = sum(cls_losses)
        reg_loss = sum(reg_losses)
        total_loss = cls_loss + self.reg_weight * reg_loss

        if self.average:
            batch = len(target_assigns)
            total_loss /= batch
            cls_loss /= batch
            reg_loss /= batch

        return total_loss, cls_loss, reg_loss

