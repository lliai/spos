import torch
import torch.nn as nn


def shuffle_channels(x, group):
    batch_size, num_channels, h, w = x.size()
    assert (num_channels % group == 0)
    x = x.view(batch_size, group, num_channels // group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, num_channels, h, w)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding):
        super(DWConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride,
                              padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)

    def forward(self, x):
        return self.bn(self.conv(x))


class ChoiceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(ChoiceBlock, self).__init__()
        self.stride = stride

        if stride == 1:
            in_ch = in_channels // 2
            out_ch = out_channels // 2
            mid_ch = in_ch
        else:
            in_ch = in_channels
            out_ch = out_channels - in_channels
            mid_ch = out_channels // 2

        self.branch1 = nn.Sequential(
            ConvBlock(in_ch, mid_ch, 1, 1, padding=0),
            DWConvBlock(mid_ch, mid_ch, ksize, stride, padding=ksize // 2),
            ConvBlock(mid_ch, out_ch, 1, 1, padding=0),
        )
        self.branch2 = nn.Sequential(
            DWConvBlock(in_ch, in_ch, ksize, stride, padding=ksize // 2),
            ConvBlock(in_ch, in_ch, 1, 1, padding=0),
        )

    def forward(self, x):
        batch_size, num_channels, h, w = x.data.size()
        if self.stride == 1:
            x1, x2 = x[:, :num_channels // 2, :, :], x[:, num_channels // 2:, :, :]
            x1 = self.branch1(x1)
        else:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        output = torch.cat([x1, x2], 1)
        output = shuffle_channels(output, group=2)

        return output


class ChoiceBlockX(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(ChoiceBlockX, self).__init__()
        self.stride = stride

        if stride == 1:
            in_ch = in_channels // 2
            out_ch = out_channels // 2
            mid_ch = in_ch
        else:
            in_ch = in_channels
            out_ch = out_channels - in_channels
            mid_ch = out_channels // 2

        self.branch1 = nn.Sequential(
            DWConvBlock(in_ch, in_ch, ksize, 1, padding=ksize // 2),
            ConvBlock(in_ch, mid_ch, 1, 1, padding=0),
            DWConvBlock(mid_ch, mid_ch, ksize, stride, padding=ksize // 2),
            ConvBlock(mid_ch, mid_ch, 1, 1, padding=0),
            DWConvBlock(mid_ch, mid_ch, ksize, 1, padding=ksize // 2),
            ConvBlock(mid_ch, out_ch, 1, 1, padding=0),
        )
        self.branch2 = nn.Sequential(
            DWConvBlock(in_ch, in_ch, ksize, stride, padding=ksize // 2),
            ConvBlock(in_ch, in_ch, 1, 1, padding=0),
        )

    def forward(self, x):
        batch_size, num_channels, h, w = x.data.size()
        if self.stride == 1:
            x1, x2 = x[:, :num_channels // 2, :, :], x[:, num_channels // 2:, :, :]
            x1 = self.branch1(x1)
        else:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        output = torch.cat([x1, x2], 1)
        output = shuffle_channels(output, group=2)

        return output
