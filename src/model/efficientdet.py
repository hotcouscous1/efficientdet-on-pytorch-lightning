from src.model.utils import *
from src.model.backbone import EfficientNet_Backbone, FeatureExtractor
from src.model.fpn import BiFPN
from src.model.head import EfficientDet_Head
from src.model.anchor import Anchor_Maker



class RetinaNet_Frame(nn.Module):

    anchor_sizes = None
    anchor_scales = None
    anchor_ratios = None
    strides = None

    def __init__(self,
                 img_size: int):

        print('The model is for images sized in {}x{}.'.format(img_size, img_size))
        super().__init__()

        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        self.backbone = None
        self.fpn = None
        self.head = None

        self.anchors = self.retinanet_anchors(img_size, self.anchor_sizes, self.anchor_scales, self.anchor_ratios, self.strides)


    def forward(self, input, detect: bool = False):
        features = self.backbone(input)
        features = self.fpn(features)
        out = self.head(features)
        if detect:
            self.detect(out)
        return out, self.anchors


    def detect(self, out):
        out[..., :2] = self.anchors[..., :2] + (out[..., :2] * self.anchors[..., 2:])
        out[..., 2:4] = torch.exp(out[..., 2:4]) * self.anchors[..., 2:]


    def retinanet_anchors(self, img_size, anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = self.retinanet_anchor_priors(anchor_sizes, anchor_scales, anchor_ratios, strides)
        anchors = Anchor_Maker(anchor_priors, strides, True, False, False)(img_size)
        return anchors


    @classmethod
    def retinanet_anchor_priors(cls, anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = []

        for stride, size in zip(strides, anchor_sizes):
            stride_priors = [[(size / stride) * s * r[0], (size / stride) * s * r[1]]
                             for s in anchor_scales
                             for r in anchor_ratios]

            anchor_priors.append(torch.Tensor(stride_priors))

        return anchor_priors




class EfficientDet(RetinaNet_Frame):

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [64, 88, 112, 160, 224, 288, 384, 384, 384],
              'head_depth':  [3, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width':  [64, 88, 112, 160, 224, 288, 384, 384, 384]}

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[1, 1], [1.4, 0.7], [0.7, 1.4]]

    strides = [8, 16, 32, 64, 128]


    def __init__(self,
                 coeff: int,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 pretrained_backbone: bool = False):

        self.img_size = self.resolutions[coeff]

        if coeff == 7:
            self.anchor_sizes = [40, 80, 160, 320, 640]

        if coeff == 8:
            self.anchor_sizes = [32, 64, 128, 256, 512, 1024]
            self.strides = [8, 16, 32, 64, 128, 256]

        num_levels = len(self.strides)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        survival_prob = self.survival_probs[coeff]

        super().__init__(self.img_size)

        if coeff < 7:
            backbone = EfficientNet_Backbone(coeff, survival_prob, nn.SiLU())
            if pretrained_backbone:
                load_pretrained(backbone, 'efficientnet_b' + str(coeff) + '_backbone')

        else:
            backbone = EfficientNet_Backbone(coeff - 1, survival_prob, nn.SiLU())
            if pretrained_backbone:
                load_pretrained(backbone, 'efficientnet_b' + str(coeff - 1) + '_backbone')

        del backbone.conv_last.layer[:]


        self.backbone = FeatureExtractor(backbone, ['stage3', 'stage5', 'stage7'])
        channels = self.backbone.widths[3: 8: 2]

        self.fpn = BiFPN(num_levels, d_bifpn, channels, w_bifpn, Act=nn.SiLU())

        self.head = EfficientDet_Head(num_levels, d_head, w_head, self.num_anchors, num_classes, nn.SiLU())

        if pretrained:
            load_pretrained(self, 'efficientdet_d' + str(coeff))


