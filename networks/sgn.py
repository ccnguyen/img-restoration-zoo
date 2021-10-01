'''
Self-Guided Network for Fast Image Denoising
Shuhang Gu, Yawei Li, Luc Van Gool, Radu Timofte
ICCV 2018
https://github.com/CurryYuan/Self-Guided-Network-for-Fast-Image-Denoising
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False, scale_factor=2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm,
                        sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm,
                        sn),
            Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation='none',
                        norm=norm, sn=sn)
        )

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = out + residual
        if self.activation:
            out = self.activation(out)
        return out


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)


class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#        Self-Guided Network (SGN)
# ----------------------------------------
class SGN(nn.Module):
    def __init__(self, in_ch, start_ch, out_ch, pad_type='zero', norm='none'):
        super(SGN, self).__init__()
        m_block = 2
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer(in_ch * (4 ** 3), start_ch * (2 ** 3), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.top2 = ResConv2dLayer(start_ch * (2 ** 3), start_ch * (2 ** 3), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.top3 = Conv2dLayer(start_ch * (2 ** 3), start_ch * (2 ** 3), 3, 1, 1, pad_type = pad_type, norm = norm)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(in_ch * (4 ** 2), start_ch * (2 ** 2), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.mid2 = Conv2dLayer(int(start_ch * (2 ** 2 + 2 ** 3 / 4)), start_ch * (2 ** 2), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.mid3 = ResConv2dLayer(start_ch * (2 ** 2), start_ch * (2 ** 2), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.mid4 = Conv2dLayer(start_ch * (2 ** 2), start_ch * (2 ** 2), 3, 1, 1, pad_type = pad_type, norm = norm)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(in_ch * (4 ** 1), start_ch * (2 ** 1), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.bot2 = Conv2dLayer(int(start_ch * (2 ** 1 + 2 ** 2 / 4)), start_ch * (2 ** 1), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.bot3 = ResConv2dLayer(start_ch * (2 ** 1), start_ch * (2 ** 1), 3, 1, 1, pad_type = pad_type, norm = norm)
        self.bot4 = Conv2dLayer(start_ch * (2 ** 1), start_ch * (2 ** 1), 3, 1, 1, pad_type = pad_type, norm = norm)
        # Mainstream
        self.main1 = Conv2dLayer(in_ch, start_ch, 3, 1, 1, pad_type = pad_type, norm = norm)
        self.main2 = Conv2dLayer(int(start_ch * (2 ** 0 + 2 ** 1 / 4)), start_ch, 3, 1, 1, pad_type = pad_type, norm = norm)
        self.main3 = nn.ModuleList([Conv2dLayer(start_ch, start_ch, 3, 1, 1, pad_type = pad_type, norm = norm)])
        self.main3.append(Conv2dLayer(start_ch, start_ch, 3, 1, 1, pad_type = pad_type, norm = norm))
        self.main3.append(Conv2dLayer(start_ch, start_ch, 3, 1, 1, pad_type = pad_type, norm = norm))
        for i in range(m_block):                            # add m conv blocks
            self.main3.append(Conv2dLayer(start_ch, start_ch, 3, 1, 1, pad_type = pad_type, norm = norm))
        self.main4 = Conv2dLayer(start_ch, out_ch, 3, 1, 1, pad_type = pad_type, norm = norm)

    def forward(self, x):
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)               # out: batch * 12 * 128 * 128
        x2 = PixelUnShuffle.pixel_unshuffle(x, 4)               # out: batch * 48 * 64 * 64
        x3 = PixelUnShuffle.pixel_unshuffle(x, 8)               # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top2(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid3(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot3(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        for model in self.main3:
            x = model(x)                                        # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        return x