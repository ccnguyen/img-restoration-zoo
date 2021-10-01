import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import summary_utils


class CharbonnierLoss(nn.Module):
	"""Charbonnier Loss (L1)"""

	def __init__(self, eps=1e-3):
		super(CharbonnierLoss, self).__init__()
		self.eps = eps

	def forward(self, x, y):
		diff = x - y
		loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
		return loss


class EdgeLoss(nn.Module):
	def __init__(self):
		super(EdgeLoss, self).__init__()
		k = torch.Tensor([[.05, .25, .4, .25, .05]])
		self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
		if torch.cuda.is_available():
			self.kernel = self.kernel.cuda()
		self.loss = CharbonnierLoss()

	def conv_gauss(self, img):
		n_channels, _, kw, kh = self.kernel.shape
		img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
		return F.conv2d(img, self.kernel, groups=n_channels)

	def laplacian_kernel(self, current):
		filtered = self.conv_gauss(current)  # filter
		down = filtered[:, :, ::2, ::2]  # downsample
		new_filter = torch.zeros_like(filtered)
		new_filter[:, :, ::2, ::2] = down * 4  # upsample
		filtered = self.conv_gauss(new_filter)  # filter
		diff = current - filtered
		return diff

	def forward(self, x, y):
		loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
		return loss


class PSNRLoss(nn.Module):
	def __init__(self, loss_weight=0.5, reduction='mean', toY=False):
		super(PSNRLoss, self).__init__()
		assert reduction == 'mean'
		self.loss_weight = loss_weight
		self.scale = 10 / np.log(10)
		self.toY = toY
		self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
		self.first = True

	def forward(self, pred, target):
		assert len(pred.size()) == 4
		if self.toY:
			if self.first:
				self.coef = self.coef.to(pred.device)
				self.first = False

			pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
			target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

			pred, target = pred / 255., target / 255.
			pass

		return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def lpips_1d(loss_fn, model_output, gt):
	model_output = torch.cat((model_output, model_output, model_output), dim=1)
	gt = torch.cat((gt, gt, gt), dim=1)
	loss = loss_fn(model_output, gt).detach().cpu().item()
	return loss


def mprnet_loss(criterion_edge, criterion_char, restored, gt):
	restored_temp = restored.repeat(1, 3, 1, 1)
	gt_temp = gt.repeat(1, 3, 1, 1)
	loss_char = criterion_char(restored_temp, gt_temp)
	loss_edge = criterion_edge(restored_temp, gt_temp)
	return loss_char + (0.05 * loss_edge)


def hinet_loss(fn, restored, gt):
	# return 1 + 0.0001 * fn(restored, gt)
	return fn(restored, gt)


def drn_loss(fn, restored, gt):
	ssim_weight = 1.1
	l1_loss_weight = 0.75
	ssim_loss = -summary_utils.get_ssim(restored, gt) * ssim_weight

	l1_loss = fn(restored, gt) * l1_loss_weight
	return l1_loss + ssim_loss


def banet_loss(fn, restored, gt):
	return 0.5 * fn(restored, gt)
