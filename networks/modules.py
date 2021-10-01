'''
Building blocks
'''
import torch.nn as nn
import torch.nn.functional as F
import torch


class MiniConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )
        self.conv_block = ConvBlock(in_ch, out_ch, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_x + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop], 1)
        return self.conv_block(out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3,
                 order='crg', num_groups=8):
        super().__init__()
        if encoder:  # generate convs for encoder path
            conv1_in = in_channels
            conv1_out = out_channels // 2
            if conv1_out < in_channels:
                conv1_out = in_channels
            conv2_in = conv1_out
            conv2_out = out_channels
        else:
            conv1_in = in_channels
            conv1_out = out_channels
            conv2_in = out_channels
            conv2_out = out_channels

        self.add_module('SingleConv1',
                        SingleConv(conv1_in, conv1_out, kernel_size, order, num_groups))
        self.add_module('SingleConv2',
                        SingleConv(conv2_in, conv2_out, kernel_size, order, num_groups))


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 order='crg', num_groups=8, padding=1):
        super().__init__()

        for name, module in self.create_conv(in_channels, out_channels,
                                             kernel_size, order,
                                             num_groups, padding):
            self.add_module(name, module)

    def create_conv(self, in_channels, out_channels, kernel_size, order, num_groups, padding=1):
        assert 'c' in order, "Conv layer MUST be present"
        assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

        modules = []
        for i, char in enumerate(order):
            if char == 'r':
                modules.append(('ReLU', nn.ReLU(inplace=True)))
            elif char == 'l':
                modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
            elif char == 'e':
                modules.append(('ELU', nn.ELU(inplace=True)))
            elif char == 'c':
                # add learnable bias only in the absence of gatchnorm/groupnorm
                bias = not ('g' in order or 'b' in order)
                modules.append(('conv', nn.conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding)))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
                # number of groups must be less or equal the number of channels
                if out_channels < num_groups:
                    num_groups = out_channels
                modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
                else:
                    modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
            else:
                raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

        return modules


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv,
                 conv_layer_order='crg', num_groups=8):
        super().__init__()
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='crg', num_groups=8):
        super().__init__()
        if basic_module == DoubleConv:
            self.upsample = None  # use neartest neighbor for upsampling
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            output_size = features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            # concatenate features with upsampled input across channel dim
            x = torch.cat((features, x), dim=1)
        else:
            # use convtranspose3d and summation joining
            x = self.upsample(x)
            x += features
        x = self.basic_module(x)
        return x
