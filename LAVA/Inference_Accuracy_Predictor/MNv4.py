import torch
import torch.nn as nn

class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class UniversalInvertedBottleneck(nn.Module):
    def __init__(self, inp, oup, start_dw_kernel, middle_dw_kernel, middle_downsample, stride, expand_ratio):
        super().__init__()
        self.stride = stride

        self.start_dw = None
        if start_dw_kernel > 0:
            self.start_dw = ConvBN(inp, inp, start_dw_kernel,
                                   stride if not middle_downsample else 1)

        hidden_dim = int(inp * expand_ratio)
        self.expand = ConvBN(inp, hidden_dim, 1, 1)

        self.middle_dw = None
        if middle_dw_kernel > 0:
            dw_stride = stride if middle_downsample else 1
            self.middle_dw = ConvBN(hidden_dim, hidden_dim, middle_dw_kernel, dw_stride)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.use_res = (stride == 1) and (inp == oup)

    def forward(self, x):
        residual = x

        if self.start_dw:
            x = self.start_dw(x)
        x = self.expand(x)
        if self.middle_dw:
            x = self.middle_dw(x)
        x = self.project(x)

        return x + residual if self.use_res else x


class MNV4ConvSmall(nn.Module):
    def __init__(self, width_mult=1.0, expand_ratio_scale=0.8):
        super().__init__()
        channels = {
            'conv0': int(32 * width_mult),
            'layer1': int(32 * width_mult),
            'layer2': int(64 * width_mult),
            'layer3': int(96 * width_mult),
            'layer4': int(128 * width_mult),
            'output': 576
        }

        expand_ratios = [
            3 * expand_ratio_scale,  # 3→2.4
            2 * expand_ratio_scale,  # 2→1.6
            4 * expand_ratio_scale  # 4→3.2
        ]

        self.features = nn.Sequential(
            # conv0
            ConvBN(3, channels['conv0'], 3, 2),

            # layer1
            ConvBN(channels['conv0'], channels['layer1'], 3, 2),
            ConvBN(channels['layer1'], channels['layer1'], 1, 1),

            # layer2
            ConvBN(channels['layer1'], channels['layer2'], 3, 2),
            ConvBN(channels['layer2'], channels['layer2'] // 2, 1, 1),

            # layer3
            UniversalInvertedBottleneck(channels['layer2'] // 2, channels['layer3'], 5, 5, True, 2, expand_ratios[0]),
            UniversalInvertedBottleneck(channels['layer3'], channels['layer3'], 0, 3, True, 1, expand_ratios[1]),

            # layer4
            UniversalInvertedBottleneck(channels['layer3'], channels['layer4'], 3, 3, True, 2, expand_ratios[2]),
            UniversalInvertedBottleneck(channels['layer4'], channels['layer4'], 0, 3, True, 1, expand_ratios[1]),

            ConvBN(channels['layer4'], channels['output'], 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.features(x).flatten(1)
