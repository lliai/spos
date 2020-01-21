import torch.nn as nn
from .choice_block import ChoiceBlock, ChoiceBlockX, ConvBlock


def select_block(in_channels, out_channels, id, stride):
    if id == 0:
        return ChoiceBlock(in_channels, out_channels, kernel_size=3, stride=stride)
    elif id == 1:
        return ChoiceBlock(in_channels, out_channels, kernel_size=5, stride=stride)
    elif id == 2:
        return ChoiceBlock(in_channels, out_channels, kernel_size=7, stride=stride)
    else:
        return ChoiceBlockX(in_channels, out_channels, kernel_size=3, stride=stride)


def create_blocks(in_channel_list, num_layer_list, arch):
    st_idx = 0
    all_blocks = []
    for layer_id, n_layers in enumerate(num_layer_list):
        block_ids = arch[st_idx:(st_idx + n_layers)]
        in_channels = in_channel_list[layer_id]
        out_channels = in_channel_list[layer_id + 1]
        all_blocks.append(select_block(in_channels, out_channels, block_ids[0], 2))
        in_channels = out_channels
        for i in range(1, n_layers):
            all_blocks.append(select_block(in_channels, out_channels, block_ids[i], 1))
        st_idx += n_layers

    return nn.Sequential(*all_blocks)


def create_supernet_blocks(in_channel_list, num_layer_list, num_block_type):
    candidates = nn.ModuleList()
    for idx, num_layer in enumerate(num_layer_list):
        in_channels = in_channel_list[idx]
        out_channels = in_channel_list[idx + 1]
        candidates.append(nn.ModuleList([select_block(in_channels, out_channels, id, 2) for id in range(num_block_type)]))
        for _ in range(1, num_layer):
            candidates.append(nn.ModuleList([select_block(out_channels, out_channels, id, 1) for id in range(num_block_type)]))

    return candidates


class SPOS(nn.Module):
    def __init__(self, in_channel_list, num_layer_list, arch, num_classes=1000):
        super(SPOS, self).__init__()
        img_channels = 3
        out_channels = 1024

        self.conv1 = ConvBlock(img_channels, in_channel_list[0], kernel_size=3, stride=2, padding=1)
        self.choice_blocks = create_blocks(in_channel_list, num_layer_list, arch)
        self.conv2 = ConvBlock(in_channel_list[-1], out_channels, kernel_size=1)
        self.global_avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(out_channels, num_classes)

        self._init_weights()

    def forward(self, x):
        out = self.choice_blocks(self.conv1(x))
        out = self.global_avgpool(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SPOS_Supernet(nn.Module):
    def __init__(self, in_channel_list, num_layer_list, num_classes=1000, num_block_type=4):
        super(SPOS_Supernet, self).__init__()
        img_channels = 3
        out_channels = 1024

        self.conv1 = ConvBlock(img_channels, in_channel_list[0], kernel_size=3, stride=2, padding=1)
        self.choice_blocks = create_supernet_blocks(in_channel_list, num_layer_list, num_block_type)
        self.conv2 = ConvBlock(in_channel_list[-1], out_channels, kernel_size=1)
        self.global_avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(out_channels, num_classes)

        self._init_weights()

    def forward(self, x, arch):
        out = self.conv1(x)
        for blk_id, l_blocks in zip(arch, self.choice_blocks):
            out = l_blocks[blk_id](out)
        out = self.global_avgpool(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
