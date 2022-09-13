from src.__init__ import *
from src.utils.bbox import iou_one_to_many


class Soft_NMS:

    def __init__(self,
                 iou_th: float = 0.3,
                 conf_th: float = 0.001,
                 per_class: bool = True,
                 gaussian: bool = True,
                 sigma: float = 0.5,
                 max_det: Optional[int] = 400,
                 bbox_format: str = 'cxcywh'):

        self.iou_th = iou_th
        self.conf_th = conf_th
        self.per_class = per_class
        self.gaussian = gaussian
        self.sigma = sigma
        self.max_det = max_det
        self.bbox_format = bbox_format


    def __call__(self, preds: Tensor) -> List[Tensor]:

        bbox_preds = preds[..., :4]
        cls_preds = preds[..., 4:]
        scores, obj_classes = torch.max(cls_preds, dim=2)

        indices = torch.arange(obj_classes.shape[1])

        pre_out = zip(bbox_preds, scores, obj_classes)
        out = []

        for pre_bbox, pre_score, pre_class in pre_out:
            idx_selected, nms_score = self.soft_nms(indices, pre_bbox, pre_score, pre_class)
            if self.max_det:
                idx_selected = idx_selected[:self.max_det]
                nms_score = nms_score[:self.max_det]

            nms_bbox = pre_bbox[idx_selected.long()]
            nms_class = pre_class[idx_selected.long()]

            nms_pred = torch.cat((nms_bbox, nms_score.view((-1, 1)), nms_class.view((-1, 1)).float()), dim=1)
            out.append(nms_pred)

        return out


    def soft_nms(self, indices, bboxes, scores, classes):
        indices = indices.clone()
        scores_remain = scores.clone()
        bboxes_remain = bboxes.clone()

        if self.per_class:
            class_weight = classes * (bboxes.max() + 1)
            bboxes_remain += class_weight.unsqueeze(dim=1)

        idx_result = torch.zeros_like(indices)
        scores_result = torch.zeros_like(scores)

        count = 0
        while scores_remain.numel() > 0:
            idx_top_score = torch.argmax(scores_remain)
            idx_result[count] = indices[idx_top_score]
            scores_result[count] = scores_remain[idx_top_score]
            count += 1

            top_box = bboxes_remain[idx_top_score]
            ious = iou_one_to_many(top_box, bboxes_remain, self.bbox_format)

            if self.gaussian:
                decay = torch.exp(-torch.pow(ious, 2) / self.sigma)
            else:
                decay = torch.ones_like(ious)
                decay_mask = ious > self.iou_th
                decay[decay_mask] = 1 - ious[decay_mask]

            scores_remain *= decay
            keep = scores_remain > self.conf_th
            keep[idx_top_score] = torch.tensor(False, device=device)

            bboxes_remain = bboxes_remain[keep]
            scores_remain = scores_remain[keep]
            indices = indices[keep]

        idx_result = idx_result[:count]
        soft_scores = scores_result[:count]

        if self.max_det:
            idx_result = idx_result[:self.max_det]
            soft_scores = soft_scores[:self.max_det]

        return idx_result, soft_scores


