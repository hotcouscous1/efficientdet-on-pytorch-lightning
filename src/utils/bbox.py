from src.__init__ import *


def iou_one_to_many(
        bbox: Tensor,
        bboxes: Tensor,
        format: str = 'cxcywh'
) -> Tensor:

    if format == 'xywh':
        x_min1 = bbox[..., 0]
        y_min1 = bbox[..., 1]
        x_max1 = bbox[..., 0] + bbox[..., 2]
        y_max1 = bbox[..., 1] + bbox[..., 3]

        x_min2 = bboxes[..., 0]
        y_min2 = bboxes[..., 1]
        x_max2 = bboxes[..., 0] + bboxes[..., 2]
        y_max2 = bboxes[..., 1] + bboxes[..., 3]

    elif format == 'cxcywh':
        x_min1 = bbox[..., 0] - bbox[..., 2] / 2
        y_min1 = bbox[..., 1] - bbox[..., 3] / 2
        x_max1 = bbox[..., 0] + bbox[..., 2] / 2
        y_max1 = bbox[..., 1] + bbox[..., 3] / 2

        x_min2 = bboxes[..., 0] - bboxes[..., 2] / 2
        y_min2 = bboxes[..., 1] - bboxes[..., 3] / 2
        x_max2 = bboxes[..., 0] + bboxes[..., 2] / 2
        y_max2 = bboxes[..., 1] + bboxes[..., 3] / 2

    elif format == 'xyxy':
        x_min1, y_min1, x_max1, y_max1 = bbox

        x_min2 = bboxes[..., 0]
        y_min2 = bboxes[..., 1]
        x_max2 = bboxes[..., 2]
        y_max2 = bboxes[..., 3]

    else:
        raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

    w_cross = (torch.min(x_max1, x_max2) - torch.max(x_min1, x_min2)).clamp(min=0).to(bbox.device)
    h_cross = (torch.min(y_max1, y_max2) - torch.max(y_min1, y_min2)).clamp(min=0).to(bbox.device)

    intersect = w_cross * h_cross

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union = area1 + area2 - intersect + 1e-7

    return intersect / union



def batch_iou(
        bboxes1: Tensor,
        bboxes2: Tensor,
        format: str = 'cxcywh'
) -> Tensor:

    if format == 'xywh':
        x_min1 = bboxes1[..., 0]
        y_min1 = bboxes1[..., 1]
        x_max1 = bboxes1[..., 0] + bboxes1[..., 2]
        y_max1 = bboxes1[..., 1] + bboxes1[..., 3]

        x_min2 = bboxes2[..., 0]
        y_min2 = bboxes2[..., 1]
        x_max2 = bboxes2[..., 0] + bboxes2[..., 2]
        y_max2 = bboxes2[..., 1] + bboxes2[..., 3]

    elif format == 'cxcywh':
        x_min1 = bboxes1[..., 0] - bboxes1[..., 2] / 2
        y_min1 = bboxes1[..., 1] - bboxes1[..., 3] / 2
        x_max1 = bboxes1[..., 0] + bboxes1[..., 2] / 2
        y_max1 = bboxes1[..., 1] + bboxes1[..., 3] / 2

        x_min2 = bboxes2[..., 0] - bboxes2[..., 2] / 2
        y_min2 = bboxes2[..., 1] - bboxes2[..., 3] / 2
        x_max2 = bboxes2[..., 0] + bboxes2[..., 2] / 2
        y_max2 = bboxes2[..., 1] + bboxes2[..., 3] / 2

    elif format == 'xyxy':
        x_min1 = bboxes1[..., 0]
        y_min1 = bboxes1[..., 1]
        x_max1 = bboxes1[..., 2]
        y_max1 = bboxes1[..., 3]

        x_min2 = bboxes2[..., 0]
        y_min2 = bboxes2[..., 1]
        x_max2 = bboxes2[..., 2]
        y_max2 = bboxes2[..., 3]

    else:
        raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

    x_min1, y_min1, x_max1, y_max1 \
        = x_min1.unsqueeze(2), y_min1.unsqueeze(2), x_max1.unsqueeze(2), y_max1.unsqueeze(2)

    x_min2, y_min2, x_max2, y_max2 \
        = x_min2.unsqueeze(1), y_min2.unsqueeze(1), x_max2.unsqueeze(1), y_max2.unsqueeze(1)

    w_cross = (torch.min(x_max1, x_max2) - torch.max(x_min1, x_min2)).clamp(min=0).to(bboxes1.device)
    h_cross = (torch.min(y_max1, y_max2) - torch.max(y_min1, y_min2)).clamp(min=0).to(bboxes1.device)

    intersect = w_cross * h_cross

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union = area1 + area2 - intersect + 1e-7

    return intersect / union



def convert_bbox(
        bbox: Tensor,
        before: str,
        after: str
) -> Tensor:

    if before == 'cxcywh':
        if after == 'xywh':
            bbox[..., 0] -= bbox[..., 2] / 2
            bbox[..., 1] -= bbox[..., 3] / 2

            return bbox

        elif after == 'xyxy':
            bbox[..., 0] -= bbox[..., 2] / 2
            bbox[..., 1] -= bbox[..., 3] / 2
            bbox[..., 2] += bbox[..., 0]
            bbox[..., 3] += bbox[..., 1]

            return bbox
        else:
            raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

    elif before == 'xywh':
        if after == 'cxcywh':
            bbox[..., 0] += bbox[..., 2] / 2
            bbox[..., 1] += bbox[..., 3] / 2

            return bbox

        elif after == 'xyxy':
            bbox[..., 2] += bbox[..., 0]
            bbox[..., 3] += bbox[..., 1]

            return bbox
        else:
            raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

    elif before == 'xyxy':
        if after == 'cxcywh':
            bbox[..., 2] = bbox[..., 2] - bbox[..., 0]
            bbox[..., 3] = bbox[..., 3] - bbox[..., 1]
            bbox[..., 0] += bbox[..., 2] / 2
            bbox[..., 1] += bbox[..., 3] / 2

            return bbox

        elif after == 'xywh':
            bbox[..., 2] -= bbox[..., 0]
            bbox[..., 3] -= bbox[..., 1]

            return bbox
        else:
            raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")
    else:
        raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")



def untransform_bbox(
        bbox: Tensor,
        scale: float,
        padding: tuple,
        format: str = 'xywh'
) -> Tensor:

    if format == 'cxcywh' or format == 'xywh':
        x = bbox[..., 0]
        y = bbox[..., 1]
        w = bbox[..., 2]
        h = bbox[..., 3]

        x /= scale
        y /= scale
        x -= padding[0]
        y -= padding[1]

        w /= scale
        h /= scale

        return bbox

    elif format == 'xyxy':
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 2]
        y_max = bbox[..., 3]

        x_min /= scale
        y_min /= scale
        x_max /= scale
        x_max /= scale

        x_min -= padding[0]
        y_min -= padding[1]
        x_max -= padding[0]
        y_max -= padding[1]

        return bbox
    else:
        raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

