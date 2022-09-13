from src.model.utils import *


class Downsampler_Pool(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 mode: str = 'maxpool',
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True
                 ):

        super().__init__()

        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

        if padding > kernel_size / 2:
            raise ValueError('pad should be smaller than half of kernel size in Pool2d')

        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')


        if mode == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)

        elif mode == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

        else:
            raise ValueError('please select the mode between maxpool and avgpool')


    def forward(self, x):
        return self.pool(x)




class FeatureFusion(nn.Module):

    def __init__(self,
                 num: int,
                 mode: str = 'sum',
                 normalize: bool = True,
                 nonlinear: nn.Module = None,
                 softmax: bool = False
                 ):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(num, dtype=torch.float32, device=device))
        self.mode = mode
        self.normalize = normalize
        self.nonlinear = nonlinear
        self.softmax = softmax


    def forward(self, features):
        weight = self.weight
        fusion = 0

        if self.nonlinear:
            weight = self.nonlinear(weight)

        if self.softmax:
            weight = weight.softmax(dim=0)

        if self.mode == 'sum':
            for w, f in zip(weight, features):
                fusion += w * f

        elif self.mode == 'mul':
            for w, f in zip(weight, features):
                fusion *= w * f

        elif self.mode == 'concat':
            features = [w * f for w, f in zip(weight, features)]
            fusion = torch.cat(features, dim=1)

        else:
            raise RuntimeError('select mode in sum, mul and concat')


        if self.normalize and not self.softmax:
            fusion /= (weight.sum() + 1e-4)

        return fusion



class Fusion(nn.Module):

    def __init__(self,
                 num: int,
                 mode: str = 'fast'
                 ):

        super().__init__()

        if mode == 'unbound':
            self.fusion = FeatureFusion(num, 'sum', normalize=False)

        elif mode == 'bound':
            self.fusion = FeatureFusion(num, 'sum', normalize=True)

        elif mode == 'softmax':
            self.fusion = FeatureFusion(num, 'sum', softmax=True)

        elif mode == 'fast':
            self.fusion = FeatureFusion(num, 'sum', normalize=True, nonlinear=nn.ReLU())

        else:
            raise ValueError('please select mode in unbound, bound, softmax, fast')

    def forward(self, features):
        return self.fusion(features)




class Resample_FPN(nn.Module):

    def __init__(self,
                 num_in: int,
                 num_out: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None
                 ):

        self.num_in, self.num_out = num_in, num_out

        if len(in_channels) != num_in:
            raise ValueError('make len(in_channels) == num_in')

        if sizes:
            if len(sizes) != num_out or len(strides) != num_out - 1:
                raise ValueError('make len(sizes) == num_out, and len(strides) == num_out - 1')


        super().__init__()

        levels = []

        for i in range(num_in):
            levels.append(Static_ConvLayer(in_channels[i], out_channels, 1, bias=True, Act=None))

        for i in range(num_in, num_out):
            if i == num_in:
                if sizes and strides:
                    levels.append(nn.Sequential(Static_ConvLayer(in_channels[-1], out_channels, 1, bias=True, Act=None),
                                                Downsampler_Pool(sizes[i - 1], sizes[i], 'maxpool', 3, strides[i - 1])))
                else:
                    levels.append(nn.Sequential(Static_ConvLayer(in_channels[-1], out_channels, 1, bias=True, Act=None),
                                                nn.MaxPool2d(3, 2, 1)))
            else:
                if sizes and strides:
                    levels.append(Downsampler_Pool(sizes[i - 1], sizes[i], 'maxpool', 3, strides[i - 1]))
                else:
                    levels.append(nn.MaxPool2d(3, 2, 1))

        self.levels = nn.ModuleList(levels)


    def forward(self, features):
        p_features = []

        for i, f in enumerate(features):
            p = self.levels[i](f)
            p_features.append(p)

        for i in range(self.num_in, self.num_out):
            f = features[-1] if i == self.num_in else p_features[-1]

            p = self.levels[i](f)
            p_features.append(p)

        return p_features



class _BiFPN(nn.Module):

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 fusion: str = 'fast',
                 Act: nn.Module = nn.SiLU()
                 ):

        self.num_levels = num_levels
        self.first = in_channels != num_levels * [out_channels]

        if sizes:
            if len(sizes) != num_levels or len(strides) != num_levels - 1:
                raise ValueError('make len(sizes) == num_levels, and len(strides) == num_levels - 1')


        super().__init__()

        if self.first:
            self.resample = Resample_FPN(len(in_channels), num_levels, in_channels, out_channels, sizes, strides)

            self.branches = nn.ModuleList([Static_ConvLayer(c, out_channels, 1, bias=True, Act=None)
                                           for c in in_channels[1: len(in_channels)]])

        if sizes:
            self.upsamples = nn.ModuleList([nn.Upsample(size=size, mode=up_mode) for size in sizes[:-1]])
        else:
            self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode=up_mode) for _ in range(num_levels - 1)])


        if sizes and strides:
            self.downsamples = nn.ModuleList([Downsampler_Pool(sizes[i], sizes[i + 1], 'maxpool', 3, strides[i])
                                             for i in range(num_levels - 1)])
        else:
            self.downsamples = nn.ModuleList([nn.MaxPool2d(3, 2, 1) for _ in range(num_levels - 1)])


        self.td_fuses = nn.ModuleList([self.fuse(2, fusion, out_channels, Act) for _ in range(num_levels - 1)])

        self.bu_fuses = nn.ModuleList([self.fuse(3, fusion, out_channels, Act) for _ in range(num_levels - 2)])
        self.bu_fuses.append(self.fuse(2, fusion, out_channels, Act))


    @staticmethod
    def fuse(num, mode, channels, Act):
        layer = [Fusion(num, mode),
                 Act,
                 Seperable_Conv2d(channels, channels, 3, 1, bias=True),
                 nn.BatchNorm2d(channels)]

        return nn.Sequential(*layer)



    def forward(self, features):
        td_features, bu_features = [], []

        # resample
        if not self.first:
            branches = features[1: -1]
        else:
            branches = []
            for i, b in enumerate(self.branches):
                branches.append(b(features[i + 1]))

            features = self.resample(features)
            branches = branches + features[len(branches) + 1: -1]


        # top-down path
        for i in range(self.num_levels - 1, -1, -1):
            if i == len(features) - 1:
                td_features.append(features[i])

            else:
                u = self.upsamples[i](td_features[-1])
                p = self.td_fuses[i]([features[i], u])

                td_features.append(p)

        td_features = td_features[::-1]


        # bottom-up path
        for i in range(self.num_levels):
            if i == 0:
                bu_features.append(td_features[i])
            else:
                d = self.downsamples[i - 1](bu_features[-1])

                if i != len(td_features) - 1:
                    p = self.bu_fuses[i - 1]([d, td_features[i], branches[i - 1]])
                else:
                    p = self.bu_fuses[i - 1]([d, td_features[i]])

                bu_features.append(p)

        return bu_features




class BiFPN(nn.Module):

    def __init__(self,
                 num_levels: int,
                 num_repeat: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 fusion: str = 'fast',
                 Act: nn.Module = nn.SiLU()
                 ):

        super().__init__()

        fpn = [_BiFPN(num_levels, in_channels, out_channels, sizes, strides, up_mode, fusion, Act)]

        for i in range(num_repeat - 1):
            fpn.append(_BiFPN(num_levels, num_levels * [out_channels], out_channels, sizes, strides, up_mode, fusion, Act))

        self.fpn = nn.ModuleList(fpn)


    def forward(self, features):
        for fpn in self.fpn:
            features = fpn(features)

        return features