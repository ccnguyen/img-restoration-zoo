import torch.nn as nn
from torch.nn import init
from functools import partial
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nets.basicblock as B

import nets.utils
"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""

# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)
        self.clamp = nets.utils.clamp_class.apply

    def forward(self, x):
        n = self.model(x)

        if x.shape[1] == 3:
            # quad exposure agg order is L,M,S
            # only do subtraction on short exposure
            img = x[:, -1:, :, :] - n
            # img = x[:, self.sub_from, :, :].unsqueeze(1) - n

        else:
            img = x - n # 1,1,H,W
        # return img
        return self.clamp(img)



def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    # msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)


def tidy_sequential(model):
    ''' Kai Zhang, 11/Jan/2019.
    '''
    for k, m in list(model.named_children()):
        if isinstance(m, nn.Sequential):
            if m.__len__() == 1:
                model._modules[k] = m.__getitem__(0)
        tidy_sequential(m)


def merge_bn(model):
    ''' Kai Zhang, 11/Jan/2019.
    merge all 'Conv+BN' (or 'TConv+BN') into 'Conv' (or 'TConv')
    based on https://github.com/pytorch/pytorch/pull/901
    '''
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        prev_m = m
        merge_bn(m)

if __name__ == '__main__':
    import torch
    model = DnCNN(in_nc=3, out_nc=1, nc=64, nb=17, act_mode='BR')

    # initialize weights
    init_weights(model,
                 init_type='orthogonal',
                 init_bn_type='uniform',
                 gain=0.2)

    x = torch.randn((1, 3, 512, 512))
    x1 = model(x)
    print(x1.shape)

