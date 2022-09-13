from src.__init__ import *
from src.utils.bbox import batch_iou



class Anchor_Maker(nn.Module):

    def __init__(self,
                 anchor_priors: Tensor or List[Tensor],
                 strides: List[int],
                 center: bool = True,
                 clamp: bool = False,
                 relative: bool = False
                 ):

        super().__init__()

        if type(anchor_priors) is Tensor or len(anchor_priors) != len(strides):
            anchor_priors = len(strides) * [anchor_priors]

        self.priors = anchor_priors
        self.strides = strides
        self.center = center
        self.clamp = clamp
        self.relative = relative


    def forward(self, img_size: int) -> Tensor:
        all_anchors = []

        for stride, priors in zip(self.strides, self.priors):
            stride_anchors = []

            num_grid = math.ceil(img_size / stride)

            if self.center:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float() + 0.5
            else:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float()

            x = grid * stride
            y = grid.t() * stride

            boxes = (stride * priors)

            for box in boxes:
                w = torch.full([num_grid, num_grid], box[0], device=device)
                h = torch.full([num_grid, num_grid], box[1], device=device)
                anchor = torch.stack((x, y, w, h))

                stride_anchors.append(anchor)

            stride_anchors = torch.cat(stride_anchors).unsqueeze(0)
            stride_anchors = stride_anchors.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

            all_anchors.append(stride_anchors)
        all_anchors = torch.cat(all_anchors, dim=1)

        if self.clamp:
            all_anchors = torch.clamp(all_anchors, 0, img_size)

        if self.relative:
            all_anchors /= img_size

        return all_anchors




class Anchor_Assigner(nn.Module):

    def __init__(self,
                 fore_th: float,
                 back_th: float = None,
                 max_for_target: bool = False,
                 foreground_only: bool = True,
                 bbox_format: str = 'cxcywh'
                 ):

        super().__init__()

        self.fore_th = fore_th
        self.back_th = back_th
        self.max_for_target = max_for_target
        self.foreground_only = foreground_only
        self.bbox_format = bbox_format


    def forward(self, labels: Tensor, anchors: Tensor) -> List[dict]:
        ious = batch_iou(anchors, labels[..., :4], self.bbox_format)
        batch_assign = []

        for i, label in enumerate(labels):

            if not (self.fore_th or self.max_for_target):
                raise ValueError("one of them must be given")

            if not self.fore_th:
                self.fore_th = 1.0 + 1e-5

            max_iou_anchor, target_for_anchor = torch.max(ious[i], dim=1)
            fore_mask = max_iou_anchor >= self.fore_th

            if self.max_for_target:
                max_iou_target, anchor_for_target = torch.max(ious[i], dim=0)
                fore_mask_target = torch.zeros(fore_mask.size(), device=device).bool()
                fore_mask_target[anchor_for_target] = True

                fore_mask = torch.logical_or(fore_mask, fore_mask_target)


            back_mask = torch.logical_not(fore_mask)

            if self.back_th:
                back_mask = torch.logical_and(back_mask, max_iou_anchor < self.back_th)
                # remainders, which are not foregrounds nor backgrounds, are invalid targets

            assigned_target = label[target_for_anchor]

            if self.foreground_only:
                batch_assign.append({'foreground': [fore_mask.nonzero(as_tuple=True)[0], assigned_target[fore_mask]]})
            else:
                batch_assign.append({'foreground': [fore_mask.nonzero(as_tuple=True)[0], assigned_target[fore_mask]],
                                     'background': [back_mask.nonzero(as_tuple=True)[0], assigned_target[back_mask]]})

        return batch_assign