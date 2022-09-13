from src.model.utils import *


class Mobile_NAS_Block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 se_ratio: float = None,
                 survival_prob: float = None):

        expand_channels = expansion * in_channels

        self.expansion, self.se_ratio, self.survival_prob = expansion, se_ratio, survival_prob
        self.shortcut = (stride == 1) and (in_channels == out_channels)

        super().__init__()

        if expansion != 1:
            self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, expand_channels),
                                          nn.BatchNorm2d(expand_channels),
                                          Act)

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(expand_channels, kernel_size, stride),
                                      nn.BatchNorm2d(expand_channels),
                                      Act)
        if se_ratio:
            self.se = Squeeze_Excitation_Conv(expand_channels, expand_channels, expansion * se_ratio, Act=Act)

        self.pw_linear = nn.Sequential(Pointwise_Conv2d(expand_channels, out_channels),
                                       nn.BatchNorm2d(out_channels))

    def forward(self, x):
        input = x
        if self.expansion != 1:
            x = self.pw_layer(x)

        x = self.dw_layer(x)
        if self.se_ratio:
            x = self.se(x)
        x = self.pw_linear(x)

        if self.shortcut:
            if self.training:
                if self.survival_prob:
                    x = stochastic_depth(x, self.survival_prob, self.training)
            x += input

        return x



class EfficientNet_Backbone(nn.Module):

    Block = Mobile_NAS_Block

    config = {'init_depth': [0, 1, 2, 2, 3, 3, 4, 1],
              'init_width': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
              'alpha': [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.6],
              'beta': [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]}


    def __init__(self,
                 coeff: int,
                 survival_prob: float = None,
                 Act: nn.Module = nn.SiLU()):

        a = self.config['alpha'][coeff]
        b = self.config['beta'][coeff]

        d = [round_depth(a * d) for d in self.config['init_depth']]
        w = [round_width(b * w, 8) for w in self.config['init_width']]

        p1, p2, p3, p4, p5, p6, p7 = get_survival_probs(d, survival_prob)

        self.coeff, self.depths, self.widths = coeff, d, w

        super().__init__()

        self.stage0 = Static_ConvLayer(3, w[0], stride=2, Act=Act)

        self.stage1 = self.Stage(d[1], w[0], w[1], 1, 3, 1, Act, 4, p1)
        self.stage2 = self.Stage(d[2], w[1], w[2], 6, 3, 2, Act, 4, p2)
        self.stage3 = self.Stage(d[3], w[2], w[3], 6, 5, 2, Act, 4, p3)
        self.stage4 = self.Stage(d[4], w[3], w[4], 6, 3, 2, Act, 4, p4)
        self.stage5 = self.Stage(d[5], w[4], w[5], 6, 5, 1, Act, 4, p5)
        self.stage6 = self.Stage(d[6], w[5], w[6], 6, 5, 2, Act, 4, p6)
        self.stage7 = self.Stage(d[7], w[6], w[7], 6, 3, 1, Act, 4, p7)

        self.conv_last = Static_ConvLayer(w[7], w[8], 1, Act=Act)


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act, se_ratio, survival_prob):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion, kernel_size, stride, Act, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion, kernel_size, 1, Act, se_ratio, survival_prob[i])

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv_last(x)

        return x



def efficientnet_b0_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(0, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b0_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b1_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(1, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b1_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b2_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(2, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b2_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b3_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(3, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b3_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b4_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(4, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b4_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b5_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(5, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b5_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b6_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(6, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b6_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b7_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(7, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b7_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model



class FeatureExtractor(nn.Module):

    def __init__(self,
                 backbone: nn.Module,
                 stages: List[str],
                 return_last: bool = False
                 ):

        super().__init__()

        self.model, self.stages, self.last = backbone, stages, return_last
        self.features = []

        if backbone.widths:
            self.widths = backbone.widths

        for name, module in self.model.named_modules():
            if name in self.stages:
                module.register_forward_hook(self.extract())


    def extract(self):
        def _extract(module, f_in, f_out):
            self.features.append(f_out)

        return _extract


    def forward(self, input):
        self.features.clear()

        if not self.last:
            _ = self.model(input)
            return self.features
        else:
            out = self.model(input)
            return self.features, out


