from src.model.utils import *


class Classifier(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU()
                 ):

        self.num_levels, self.num_anchors, self.num_classes \
            = num_levels, num_anchors, num_classes

        super().__init__()

        self.conv_layers = nn.ModuleList([Seperable_Conv2d(width, width, 3, 1, bias=True)
                                          for _ in range(depth)])

        self.bn_layers = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
                                        for _ in range(num_levels)])
        self.act = Act

        self.conv_pred = Seperable_Conv2d(width, num_anchors * num_classes, bias=True)


    def forward(self, features):
        out = []
        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], pred.shape[1], pred.shape[2], self.num_anchors, self.num_classes)
            pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class Regressor(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 Act: nn.Module = nn.SiLU()
                 ):

        self.num_levels = num_levels

        super().__init__()

        self.conv_layers = nn.ModuleList([Seperable_Conv2d(width, width, 3, 1, bias=True)
                                          for _ in range(depth)])

        self.bn_layers = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
                                        for _ in range(num_levels)])
        self.act = Act

        self.conv_pred = Seperable_Conv2d(width, num_anchors * 4, bias=True)


    def forward(self, features):
        out = []
        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1).contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class EfficientDet_Head(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU()
                 ):

        super().__init__()

        self.classifier = Classifier(num_levels, depth, width, num_anchors, num_classes, Act)
        self.regressor = Regressor(num_levels, depth, width, num_anchors, Act)


    def forward(self, features):
        reg_out = self.regressor(features)
        cls_out = self.classifier(features)
        out = torch.cat((reg_out, cls_out), dim=2)
        return out
