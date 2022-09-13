from src.__init__ import *


def Depthwise_Conv2d(
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False
):
    if not padding and padding != 0:
        padding = dilation * (kernel_size - 1) // 2

    dw_conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                        groups=channels, bias=bias)
    return dw_conv



def Pointwise_Conv2d(
        in_channels: int,
        out_channels: int,
        bias: bool = False
):
    pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
    return pw_conv



def Seperable_Conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False
):
    conv = nn.Sequential(Depthwise_Conv2d(in_channels, kernel_size, stride, padding, dilation, False),
                         Pointwise_Conv2d(in_channels, out_channels, bias))
    return conv



class Static_ConvLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = False,
                 batch_norm: bool = True,
                 Act: None or nn.Module = nn.ReLU(inplace=False),
                 **kwargs
                 ):

        batch_eps = kwargs.get('eps', 1e-05)
        batch_momentum = kwargs.get('momentum', 0.1)

        padding = (kernel_size - 1) // 2


        super().__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]

        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
        if Act:
            layer.append(Act)

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)




class Dynamic_ConvLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 batch_norm: bool = True,
                 Act: None or nn.Module = nn.ReLU(inplace=False),
                 reverse: str = None,
                 **kwargs
                 ):

        batch_eps = kwargs.get('eps', 1e-05)
        batch_momentum = kwargs.get('momentum', 0.1)

        if not padding and padding != 0:
            padding = dilation * (kernel_size - 1) // 2


        super().__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]

        if not reverse:
            if batch_norm:
                layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.append(Act)

        elif reverse == 'ACB':
            if batch_norm:
                layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(0, Act)

        elif reverse == 'BAC':
            if batch_norm:
                layer.insert(0, nn.BatchNorm2d(in_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(-1, Act)

        elif reverse == 'ABC':
            if batch_norm:
                layer.insert(0, nn.BatchNorm2d(in_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(0, Act)

        else:
            raise ValueError('reverse order should be one of ACB, BAC, ABC')

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)



class Squeeze_Excitation_Conv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 reduction: float,
                 batch_norm: bool = False,
                 Act: nn.Module = nn.ReLU(),
                 Sigmoid: nn.Module = nn.Sigmoid(),
                 **kwargs
                 ):

        divisor = kwargs.get('divisor', 1)
        round_bias = kwargs.get('round_bias', 0.9)

        reduct_channels = round_width(channels // reduction, divisor, round_bias)


        super().__init__()

        squeeze = [nn.AdaptiveAvgPool2d(1),
                   nn.Conv2d(in_channels, reduct_channels, kernel_size=1)]

        if not batch_norm:
            squeeze.append(Act)
        else:
            squeeze.append(nn.BatchNorm2d(reduct_channels))
            squeeze.append(Act)

        excitation = [nn.Conv2d(reduct_channels, channels, kernel_size=1),
                      Sigmoid]

        self.squeeze = nn.Sequential(*squeeze)
        self.excitation = nn.Sequential(*excitation)


    def forward(self, input):
        x = self.squeeze(input)
        x = self.excitation(x)
        x = x * input
        return x



def stochastic_depth(
        input: Tensor,
        survival_prob: float,
        training: bool
):

    if not training:
        raise RuntimeError('only while training, drop connect can be applied')

    batch_size = input.shape[0]
    random_mask = survival_prob + torch.rand([batch_size, 1, 1, 1], device=input.device)
    binary_mask = torch.floor(random_mask)

    output = input / survival_prob * binary_mask
    return output



def get_survival_probs(
        num_block_list:list,
        last_survival_prob: float
):

    num_blocks = sum(num_block_list[1:])

    survival_probs = []
    for num, end in zip(num_block_list[1:], itertools.accumulate(num_block_list[1:])):
        if last_survival_prob:
            survival_probs.append(
                1 - (torch.Tensor(range(end-num, end)) / num_blocks) * (1 - last_survival_prob))
        else:
            survival_probs.append([None for _ in range(end-num, end)])

    return survival_probs



def make_divisible(
        value: int or float,
        divisor: int = 8,
        round_bias: float = 0.9
):
    round_value = max(divisor, int(value + divisor / 2) // divisor * divisor)

    assert 0 < round_bias < 1
    if round_value < round_bias * value:
        round_value += divisor

    return round_value



def round_width(
        width: int or float,
        divisor: int = 8,
        round_bias: float = 0.9
):
    return make_divisible(width, divisor, round_bias)



def round_depth(depth: int):
    return math.ceil(depth)



def load_pretrained(
        model: nn.Module,
        ckpt_name: str,
        strict: bool = True,
        batch_eps: float = None,
        batch_momentum: float = None
):

    if ckpt_name not in checkpoints:
        raise ValueError('<Sorry, checkpoints for ' + ckpt_name + ' is not ready>')

    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoints[ckpt_name], map_location=device), strict)
    print('<All keys matched successfully>')

    if batch_eps or batch_momentum:
        batch_params(model, batch_eps, batch_momentum)



def batch_params(
        module: nn.Module,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
):

    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = eps
            m.momentum = momentum

            if not affine:
                m.affine = affine

            if not track_running_stats:
                m.track_running_stats = track_running_stats

