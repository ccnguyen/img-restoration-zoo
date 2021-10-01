import numpy as np
import torch
import torchvision.utils
from skimage.metrics import structural_similarity as ssim


def get_psnr(pred, gt, mask=None):
    if mask is None:
        return 10 * torch.log10(1 / torch.mean((pred - gt) ** 2)).detach().cpu().numpy()
    else:
        mse = mask * (pred - gt) ** 2
        mse = mse.sum() / mask.sum()
        return 10 * torch.log10(1 / mse).detach().cpu().numpy()


def get_ssim(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        ssims.append(ssim(pred_i, gt_i, multichannel=True))
    return sum(ssims) / len(ssims)


def write_val_scalars(writer, names, values, total_steps):
    for name, val in zip(names, values):
        writer.add_scalar(f'val/{name}', np.mean(val), total_steps)



def write_summary(batch_size, writer, gt, output, total_steps, optim):
    result_gt = torch.cat((output, gt), dim=0)
    grid = torchvision.utils.make_grid(result_gt,
                                       scale_each=True,
                                       nrow=batch_size,
                                       normalize=False).cpu().detach().numpy()

    psnr = get_psnr(output, gt)
    ssim = get_ssim(output, gt)
    writer.add_image(f"avg_result_gt", grid, total_steps)
    writer.add_scalar(f"train/psnr", psnr, total_steps)
    writer.add_scalar(f"train/ssim", ssim, total_steps)
    writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], total_steps)
