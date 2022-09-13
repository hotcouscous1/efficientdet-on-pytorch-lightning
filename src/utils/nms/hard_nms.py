from src.__init__ import *
from src.utils.bbox import iou_one_to_many, convert_bbox
from torchvision.ops import batched_nms



class Hard_NMS:

    def __init__(self,
                 iou_th: float = 0.5,
                 max_det: Optional[int] = 400,
                 bbox_format: str = 'cxcywh'):

        self.iou_th = iou_th
        self.max_det = max_det
        self.bbox_format = bbox_format


    def __call__(self, preds: Tensor) -> List[Tensor]:
        bbox_preds = preds[..., :4]
        cls_preds = preds[..., 4:]
        scores, obj_classes = torch.max(cls_preds, dim=2)

        if self.bbox_format != 'xyxy':
            bbox_preds = convert_bbox(bbox_preds, self.bbox_format, 'xyxy')

        pre_out = zip(bbox_preds, scores, obj_classes)
        out = []

        for pre_bbox, pre_score, pre_class in pre_out:
            idx_selected = batched_nms(pre_bbox, pre_score, pre_class, self.iou_th)
            if self.max_det:
                idx_selected = idx_selected[:self.max_det]

            nms_bbox = pre_bbox[idx_selected]
            nms_score = pre_score[idx_selected]
            nms_class = pre_class[idx_selected]

            if self.bbox_format != 'xyxy':
                nms_bbox = convert_bbox(nms_bbox, 'xyxy', self.bbox_format)

            nms_pred = torch.cat((nms_bbox, nms_score.view((-1, 1)), nms_class.view((-1, 1)).float()), dim=1)
            out.append(nms_pred)

        return out



class Yolo_NMS:

    def __init__(self,
                 iou_th: float = 0.5,
                 conf_th: float = 0.001,
                 max_det: Optional[int] = 400,
                 bbox_format: str = 'cxcywh'):

        self.iou_th = iou_th
        self.conf_th = conf_th
        self.max_det = max_det
        self.bbox_format = bbox_format


    def __call__(self, preds: Tensor) -> List[Tensor]:

        bbox_preds = preds[..., :4]
        cls_preds = preds[..., 4:]
        scores, obj_classes = torch.max(cls_preds, dim=2)

        pre_out = zip(bbox_preds, scores, obj_classes)
        out = []

        for pre_bbox, pre_score, pre_class in pre_out:
            idx_selected = self.yolo_nms(pre_bbox, pre_score, pre_class)
            if self.max_det:
                idx_selected = idx_selected[:self.max_det]

            nms_bbox = pre_bbox[idx_selected]
            nms_score = pre_score[idx_selected]
            nms_class = pre_class[idx_selected]

            nms_pred = torch.cat((nms_bbox, nms_score.view((-1, 1)), nms_class.view((-1, 1)).float()), dim=1)
            out.append(nms_pred)

        return out


    @classmethod
    def group_by_class(cls, classes):
        num_classes = classes.shape[-1]
        class_group = [[] for _ in range(num_classes)]

        for i, pred in enumerate(classes):
            class_group[int(torch.argmax(pred))].append(i)

        return class_group


    def yolo_nms(self, bboxes, scores, classes):
        indices = bboxes.shape[0]
        if indices == 0:
            return bboxes, scores, classes

        conf_index = torch.nonzero(torch.ge(scores, self.conf_th)).squeeze()

        bboxes = bboxes[conf_index]
        scores = scores[conf_index]
        classes = classes[conf_index]

        idx_class_group = self.group_by_class(classes)
        idx_result = []

        for class_, idx_member in enumerate(idx_class_group):
            idx_member = torch.Tensor(idx_member).long()
            bbox_class = bboxes[idx_member]
            score_class = scores[idx_member]

            score_class, idx_remain = torch.sort(score_class, descending=False)

            idx_class_result = []
            while idx_remain.size(0) != 0:
                idx_selected = idx_remain[-1]
                idx_class_result.append(idx_selected)

                bbox_selected = bbox_class[idx_selected]
                ious = iou_one_to_many(bbox_selected, bbox_class[idx_remain[:-1]], self.bbox_format)
                keep = torch.nonzero(ious <= self.iou_th).squeeze()

                idx_remain = idx_remain.index_select(dim=0, index=keep)

            idx_result.extend([idx_member[i] for i in idx_class_result])
        idx_result = bboxes.new_tensor(idx_result, dtype=torch.long)

        return idx_result

