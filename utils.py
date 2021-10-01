import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import random
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets.scheduler import GradualWarmupScheduler
from nets.banet import LinearDecay
import nets.loss as loss


def seed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def assign_p():
    p = psutil.Process()
    cores = [i for i in range(27)]
    p.cpu_affinity(cores)


def modify_args(args):
    if args.decoder == 'unet':
        args.loss = 'l2'
    if args.local:
        args.remote = False
    else:
        args.remote = True

    args.date_resume = args.resume
    if not hasattr(args, 'shutter'):
        args.shutter = None
    if args.date_resume != '00-00':
        date = f'2021-{args.date_resume}'
    else:
        date = datetime.date.today().strftime('%y-%m-%d')

    if args.test:
        exp_name = f'{args.log_root}/test'
        args.steps_til_summary = 10
    else:
        dir_name = f'{args.log_root}/{date}/{date}-{args.decoder}'
        os.makedirs(dir_name, exist_ok=True)
        exp_name = f'{dir_name}/{get_exp_name(args)}'

        if args.date_resume != '00-00' and not os.path.exists(exp_name):
            raise ValueError('This directory does not exist :-(')

    return args, exp_name


def save_best(args, total_steps, epoch, val_psnrs, val_ssims, val_lpips, checkpoints_dir):
    single_column_names = ['Model', 'Total Steps', 'Epoch', 'PSNR', 'SSIM', 'LPIPS']

    df = pd.DataFrame(columns=single_column_names)
    series = pd.Series([args.decoder,
                        total_steps,
                        epoch,
                        round(np.mean(val_psnrs), 3),
                        round(np.mean(val_ssims), 3),
                        round(np.mean(val_lpips), 3)],
                       index=df.columns)
    df = df.append(series, ignore_index=True)
    file_name = f'{checkpoints_dir}/val_results.csv'
    # overwrite the file and just save the best recent
    df.to_csv(file_name, header=single_column_names)


def define_loss(args):
    if args.decoder == 'mpr' or args.decoder == 'uform':
        char_loss = loss.CharbonnierLoss()
        edge_loss = loss.EdgeLoss()
        loss_fn = partial(loss.mprnet_loss, edge_loss, char_loss)
    elif args.decoder == 'hi':
        psnr_loss = loss.PSNRLoss()
        loss_fn = partial(loss.hinet_loss, psnr_loss)
    elif args.decoder == 'banet':
        l2_loss = nn.MSELoss()
        loss_fn = partial(loss.banet_loss, l2_loss)
    elif args.decoder == 'drn':
        l1_loss = nn.MSELoss()
        loss_fn = partial(loss.drn_loss, l1_loss)
    elif args.loss == 'l2':
        l2_loss = nn.MSELoss()
        loss_fn = partial(l2_loss)
    else:
        # dncnn uses l1 loss
        l1_loss = nn.L1Loss()
        loss_fn = partial(l1_loss)
    return loss_fn

def make_model_dir(exp_name, restart=False):
    version_num = find_version_number(exp_name)
    model_dir = f'{exp_name}/v_{version_num}'
    if restart:
        model_dir = f'{exp_name}/v_{version_num-1}'
        # remove the current dir and make a new one
        if os.path.exists(model_dir):
            os.remove(model_dir)
    print(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    return model_dir, version_num



def get_date(date_resume):
    if date_resume != '00-00':
        return f'2021-{date_resume}'
    else:
        return datetime.date.today().strftime('%Y-%m-%d')


def define_schedule(optim, args):
    if args.decoder == 'dncnn' or args.decoder == 'dncnn_n':
        args.sched = 'multi'
        milestones = [660*200, 660*600, 660*1200, 660*2400, 660*4800]
    if args.decoder in ['srgan', 'unet']:
        milestones = [660*50, 660*200, 660*400, 660*800, 660*1600]
    if args.decoder == 'banet':
        args.sched = 'linear'
    if args.sched == 'cosine':
        warmup_epochs = 3
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                                      args.max_epochs - warmup_epochs,
                                                                      eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optim, multiplier=1,
                                           total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
        return scheduler
    elif args.sched == 'reducelr':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                          factor=0.7,
                                                          patience=10,
                                                          threshold=1e-4,
                                                          threshold_mode='rel',
                                                          cooldown=0,
                                                          min_lr=2e-5,
                                                          eps=1e-08,
                                                          verbose=False)
    elif args.sched == 'warm':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=args.s_iters, T_mult=1, eta_min=0,
                                                                    last_epoch=-1, verbose=False)
    elif args.sched == 'multi':
        return torch.optim.lr_scheduler.MultiStepLR(optim,
                                                    milestones=milestones,
                                                    gamma=0.5)
    elif args.sched == 'linear':
        return LinearDecay(optim,
                                    min_lr=1e-7,
                                    num_epochs=args.max_epochs,
                                    start_epoch=50)
    return None


def define_optim(model, args):
    return torch.optim.AdamW([{'params': model.parameters(), 'lr': args.lr}], lr=2e-4, eps=1e-2)


def save_chkpt(model, optim, checkpoints_dir, epoch=0, val_psnrs=None, final=False, best=False):
    if best:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_best.pth'))
        return
    if final:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'val_psnrs_final.txt'),
                   np.array(val_psnrs))
        return
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
        if val_psnrs is not None:
            np.savetxt(os.path.join(checkpoints_dir, 'train_psnrs_epoch_%04d.txt' % epoch),
                       np.array(val_psnrs))
        return


def convert(img, dim=1):
    if dim == 1:
        return img.squeeze(0).squeeze(0).detach().cpu().numpy()
    if dim == 3:
        return img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


### functions for keeping train script clean ####
def make_subdirs(model_dir, make_dirs=True):
    summaries_dir = f'{model_dir}/summaries'
    checkpoints_dir = f'{model_dir}/checkpoints'
    if make_dirs:
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
    return summaries_dir, checkpoints_dir


def find_version_number(path):
    if not os.path.isdir(path):
        return 0
    fnames = sorted(os.listdir(path))
    latest = fnames[-1]
    latest = latest.rsplit('_', 1)[-1]
    return int(latest) + 1


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_quiver(gt, displacement, count, fname):
    sz = gt.shape[-1]
    temp = displacement.view(-1, 2).cpu().detach()
    dx = temp[:, 0].view(sz, sz).numpy()
    dy = temp[:, 1].view(sz, sz).numpy()
    x, y = np.meshgrid(np.arange(gt.shape[-1]), np.arange(gt.shape[-1]))
    skip = 25
    plt.figure()
    plt.imshow(convert(gt, dim=1), cmap='gray')
    plt.quiver(x[::skip, ::skip], y[::skip, ::skip], dx[::skip, ::skip], dy[::skip, ::skip], color='yellow')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{fname}/{count}_quiver.png')


def get_exp_name(args):
    ''' Make folder name readable '''
    printedargs = ''
    forbidden = ['data_root', 'log_root', 'block_size', 'test',
                 'remote', 'max_epochs', 'batch_size', 'num_workers',
                 'steps_til_summary', 'epochs_til_checkpoint', 'num_freq', 'nl',
                 'last_act', 'date_resume', 'list', 'reg', 's_iters',
                 'steps_til_ckpt', 'merge_bn', 'merge_bn_startpoint',
                 'sub_from', 'restart', 'no_neptune', 'slr', 'resume', 'no_cudnn', 'ares']
    for k, v in vars(args).items():
        if k not in forbidden:
            print(f'{k} = {v}')
            if k == 'sig_shot' and v < 0:
                continue
            if k == 'noise':
                k = 'n'
            if k == 'decoder':
                k = 'dec'
            if k == 'shutter':
                k = 'shut'
            printedargs += f'{k}={v}_'
    return printedargs


def augmentData(imgs):
    aug = random.randint(0, 8)

    num = len(imgs)
    # Data Augmentations
    if aug == 1:
        for i in range(num):
            imgs[i] = imgs[i].flip(-2)
    elif aug == 2:
        for i in range(num):
            imgs[i] = imgs[i].flip(-1)
    elif aug == 3:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1))
    elif aug == 4:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=2)
    elif aug == 5:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=3)
    elif aug == 6:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-2), dims=(-2, -1))
    elif aug == 7:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-1), dims=(-2, -1))
    return imgs
