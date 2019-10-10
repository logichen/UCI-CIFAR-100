import math

import torch
from torch import nn
from torch.autograd import Variable

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1, bias=True),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_planes == out_planes and stride == 1)
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1),
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand((batch_size, 1, 1, 1), dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

####### arcloss ##############
def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class L2norm(nn.Module):
    def __init__(self, axis=1):
        super(L2norm, self).__init__()
        self.axis = axis
    def forward(self, input,):
        norm = torch.norm(input,2, self.axis,True)
        output = torch.div(input, norm)
        return output

class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=200, classnum=100):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
    def forward(self, embbedings):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        # print(cos_theta)
        return cos_theta

def Arctransf(feat, s=5.):
    output = feat*1.0
    output *= s    
    return output

class Margloss(nn.Module):  
    def __init__(self,  s=5.):
        super(Margloss, self).__init__()
        self.s = s 
    def forward(self, feat, label):
        nB = len(feat)
        output = feat * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        loss = nn.CrossEntropyLoss()
        out = loss(output, label)
        return out   # return out--loss

###############################

class EfficientNet(nn.Module):

    LR_REGIME = [1, 25, 0.05, 26, 50, 0.005, 51, 65, 0.0005, 66, 70, 0.00005]

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=100, coslinear=True):
        super(EfficientNet, self).__init__()

        # yapf: disable
        settings = [

            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE,  32 ->  32
            [6,  24, 2, 1, 3],  # MBConv6_3x3, SE,  32 ->  32   
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  32 ->  16   
            [6,  80, 3, 1, 3],  # MBConv6_3x3, SE,  16 ->  16   
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  16 ->   8   16->16
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,   8 ->   8   16->8
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   8 ->   8   8->8
            ####

        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=1)]
        ##### 

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(in_channels, last_channels, 1)]

        self.features = nn.Sequential(*features)
        self.pool = nn.AvgPool2d(8)
        if coslinear:
            print('using coslinear')
            mid_channels = 200
            self.classifier = nn.Sequential(
                # nn.Dropout(dropout_rate),
                nn.Linear(last_channels, mid_channels),
                L2norm(),
                # nn.Dropout(dropout_rate),
                Arcface(mid_channels, num_classes),
            )
        else:
            print('using Linear')
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(last_channels, num_classes),
            )


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # x = x.mean([2, 3])
        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)

def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)

