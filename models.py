import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets.nets import UNet
from nets.mprnet import MPRNet_s2
from nets.hinet import HINet
from nets.dncnn import DnCNN, init_weights
from nets.sgn import SGN
from nets.uformer import Uformer
from nets.drn import DRN
from nets.banet import BANet_model
from nets.msrn import MSRN


def define_decoder(model_name, args):
	in_ch = 1
	if model_name == 'msrn':
		return MSRN(in_ch=in_ch, out_ch=1)

	if model_name == 'uform':
		depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
		input_size = args.block_size[-1]
		return Uformer(img_size=input_size, embed_dim=44, depths=depths,
                       se_layer=False, in_chans=in_ch, out_chans=1)
	if model_name == 'unet':
		return UNet(in_ch, out_ch=1, depth=6,
                        growth=5,
                        padding=True,
                        batch_norm=False)
	if model_name == 'mpr':
		return MPRNet_s2(in_c=in_ch)
	if model_name == 'hi':
		return HINet(in_ch=in_ch)
	if model_name == 'sgn':
		return SGN(in_ch=in_ch, out_ch=in_ch)
	if model_name == 'drn':
		return DRN(in_ch=in_ch, out_ch=1)
	if model_name == 'banet':
		return BANet_model(in_ch=in_ch, out_ch=1)

	if model_name == 'dncnn':
		model = DnCNN(in_nc=in_ch, out_nc=1, nc=64, nb=17, act_mode='BR')
		init_weights(model,
                     init_type='orthogonal',
                     init_bn_type='uniform',
                     gain=0.2)
        return model
