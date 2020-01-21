import torch
import torch.nn as nn
import models
from thop import profile
from thop.count_hooks import zero_ops


class SPOS_Supernet_FLOPs(nn.Module):
    def __init__(self, in_channel_list, num_layer_list, num_classes=1000, num_block_type=4):
        super(SPOS_Supernet_FLOPs, self).__init__()
        img_channels = 3
        out_channels = 1024
        self.in_channel_list = in_channel_list
        self.img_size = [112, 56, 28, 14, 7]

        self.conv1 = models.ConvBlock(img_channels, in_channel_list[0], kernel_size=3, stride=2, padding=1)
        self.choice_blocks = models.create_supernet_blocks(in_channel_list, num_layer_list, num_block_type)
        self.conv2 = models.ConvBlock(in_channel_list[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.global_avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x, idx):
        if idx == 0:
            return self.conv1(x)
        elif idx == 2:
            return self.conv2(x)
        elif idx == 3:
            return self.global_avgpool(x)
        elif idx == 4:
            return self.fc(x)
        else:
            layer_idx, block_idx = idx
            return self.choice_blocks[layer_idx][block_idx](x)


def count_flops(model, inputs):
    if not isinstance(inputs, tuple):
        inputs = (inputs, )
    flops, params = profile(model, inputs=inputs, custom_ops={nn.BatchNorm2d: zero_ops}, verbose=False)
    return flops


def write_flops(save_path):
    img_size = [112, 56, 28, 14, 7]
    in_channel_list = [16, 64, 160, 320, 640]
    num_layer_list = [4, 4, 8, 4]
    Net = SPOS_Supernet_FLOPs(in_channel_list, num_layer_list)

    total_flops = []
    for i in range(5):
        if i == 1:
            for layer_idx in range(20):
                if layer_idx == 0:
                    x = torch.randn(1, in_channel_list[0], img_size[0], img_size[0])
                elif layer_idx <= 4:
                    x = torch.randn(1, in_channel_list[1], img_size[1], img_size[1])
                elif layer_idx <= 8:
                    x = torch.randn(1, in_channel_list[2], img_size[2], img_size[2])
                elif layer_idx <= 16:
                    x = torch.randn(1, in_channel_list[3], img_size[3], img_size[3])
                else:
                    x = torch.randn(1, in_channel_list[-1], img_size[-1], img_size[-1])

                cur_flops = []
                for block_idx in range(4):
                    idx = [layer_idx, block_idx]
                    flops = count_flops(Net, (x, idx))
                    cur_flops.append(flops)
                total_flops.append(cur_flops)
        else:
            if i == 0:
                x = torch.randn(1, 3, 224, 224)
            elif i == 2:
                x = torch.randn(1, in_channel_list[-1], img_size[-1], img_size[-1])
            elif i == 3:
                x = torch.randn(1, 1024, img_size[-1], img_size[-1])
            else:
                x = torch.randn(1, 1024)

            flops = count_flops(Net, (x, i))
            total_flops.append([flops] * 4)

    torch.save(total_flops, save_path)


def get_flops(arch, flop_table):
    total_flops = 0
    for i, flops in enumerate(flop_table):
        if i == 0 or i >= 21:
            total_flops += flops[0]
        else:
            layer_idx = i - 1
            block_idx = arch[layer_idx]
            total_flops += flops[block_idx]

    return total_flops
