"""TwoStream architecture with ResNet50 encoder

The architecture takes a 6-channel input (two images), feeds each half (one image each) into a ResNet50 encoder network. Intermediate feature maps are kept. The output before the fully connected layer of the encoders are differenced and fed into a bridge and then subsequently upsample within a decoder network. Via skip connections high resolution feature maps are passed to the decoder, each being a difference of the intermediate maps of the two encoders. Nearest neighbour upsampling is used.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
import copy

class TwoStream_Resnet50_Diff(nn.Module):
    """TwoStream ResNet50 with difference skip connections
    


    """    
    def __init__(
        self,
        in_channels=6,
        n_classes=5,
        seperate_loss = False,
        pretrained = True,
        output_features = False,
        shared = False,
        diff = True
        ):
        """Initialize model
        
        Args:
            in_channels (int, optional): Number of input channels, please keep!. Defaults to 6.
            n_classes (int, optional): Number of classes. Defaults to 5.
            seperate_loss (bool, optional): If True, two seperate heads for localization and damage predictions are used. Defaults to False.
            pretrained (bool, optional): If True uses ImageNet pretrained weights for the two ResNet50 encoders. Defaults to True.
            output_features (bool, optional): If True forward pass outputs feature maps after bridge. Defaults to False
            shared (bool, optional): If True, model is siamese (shared encoder). Defaults to False.
            diff (bool, optional): If True, difference is fed to decoder, else an intermediate 1x1 conv merges the two downstream features. Defaults to True.
        """        

        super(TwoStream_Resnet50_Diff, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        self.seperate_loss = seperate_loss
        self.output_features = output_features
        self.shared = shared
        self.diff = diff
        down_blocks1 = []
        up_blocks = []
        self.input_block1 = copy.deepcopy(nn.Sequential(*list(self.resnet.children()))[:3])
        if not shared:
            down_blocks2 = []
            self.input_block2 = copy.deepcopy(nn.Sequential(*list(self.resnet.children()))[:3])
        self.input_pool = copy.deepcopy(list(self.resnet.children())[3])
        for bottleneck in list(self.resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks1.append(copy.deepcopy(bottleneck))
                if not shared:
                    down_blocks2.append(copy.deepcopy(bottleneck))
        self.down_blocks1 = nn.ModuleList(down_blocks1)
        if not shared:
            self.down_blocks2 = nn.ModuleList(down_blocks2)
        self.bridge = Bridge(2048, 2048)

        if not diff:
            self.concat_blocks = nn.ModuleList([nn.Conv2d(n, int(n/2), kernel_size=1, stride = 1) for n in [4096,2048,1024,512,128,6]])

        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        if self.seperate_loss:
            self.out_loc = nn.Conv2d(64, 2, kernel_size=1, stride=1)
            self.out_dmg = nn.Conv2d(64, 5, kernel_size=1, stride=1)
        else:
            self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        del self.resnet


    def forward(self, x, return_features = None):
        """Forward pass
        
        Args:
            x (torch.Tensor): The 6-channel input
        
        Returns:
            torch.Tensor: A 5-channel output activations map
        """        
        x1, x2 = torch.split(x, 3, dim = 1)
        del x
        pre_pools1 = dict()
        pre_pools1[f"layer_0"] = x1
        x1 = self.input_block1(x1)
        pre_pools1[f"layer_1"] = x1
        x1 = self.input_pool(x1)

        for i, block in enumerate(self.down_blocks1, 2):
            x1 = block(x1)
            if i == (5):
                continue
            pre_pools1[f"layer_{i}"] = x1

        pre_pools2 = dict()
        pre_pools2[f"layer_0"] = x2
        if not self.shared:
            x2 = self.input_block2(x2)
        else:
            x2 = self.input_block1(x2)
        pre_pools2[f"layer_1"] = x2
        x2 = self.input_pool(x2)

        if not self.shared:
            tmp_down = self.down_blocks2
        else:
            tmp_down = self.down_blocks1
        for i, block in enumerate(tmp_down, 2):
            x2 = block(x2)
            if i == (5):
                continue
            pre_pools2[f"layer_{i}"] = x2

        if self.diff:
            x = torch.add(x1, -1, x2)
        else:
            x = self.concat_blocks[0](torch.cat([x1,x2],1))
        x = self.bridge(x)
        if self.output_features:
            features = x

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{5 - i}"
            if self.diff:
                skip = torch.add(pre_pools1[key], -1, pre_pools2[key])
            else:
                skip = self.concat_blocks[i](torch.cat([pre_pools1[key],pre_pools2[key]],1))
            x = block(x, skip)
        del pre_pools1, pre_pools2, skip
        if self.seperate_loss:
            return self.out_loc(x),self.out_dmg(x)
        else:
            if self.output_features:
                return self.out(x), features
            elif return_features:
                return self.out(x), x
            else:
                return self.out(x)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="nearest"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "nearest":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=2),
                nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
