import lpips
import os
import sys
import time
from argparse import ArgumentParser
from functools import partial

import json
import numpy as np
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

# Enable import from parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nets
import models
import utils
import summary_utils
import dataloading

utils.seed(1234)


def main(args):
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    custom_loss_fn = partial(nets.loss.lpips_1d, loss_fn)
    args, exp_name = utils.modify_args(args)
    model = models.define_decoder(args.decoder, args)

    model_dir, version_num = utils.make_model_dir(exp_name)
    model.cuda()

    summaries_dir, checkpoints_dir = utils.make_subdirs(model_dir)
    writer = SummaryWriter(summaries_dir)

    if args.date_resume != '00-00':
        file = [f for f in os.listdir(f'{exp_name}/v_{version_num-1}/checkpoints') if 'current.pth' in f][0]

        model.load_state_dict(torch.load(
            f'{exp_name}/v_{version_num - 1}/checkpoints/{file}'))
    else:
        if not args.test:
            with open(f'{exp_name}/args.json', 'w') as f:
                json.dump(vars(args), f, indent=4)

    train_dataloader = dataloading.loadTrainingDataset(args)
    val_dataloader = dataloading.loadValDataset(args)

    summary_fn = partial(summary_utils.write_summary, args.batch_size, writer)

    optim = utils.define_optim(model, args)

    scheduler = utils.define_schedule(optim, args)
    loss_fn = utils.define_loss(args)
    total_time_start = time.time()

    best_val_psnr = 0
    total_steps = 0
    all_val_psnrs = []

    with tqdm(total=len(train_dataloader) * args.max_epochs) as pbar:
        for epoch in range(args.max_epochs):
            if not epoch % args.epochs_til_checkpoint and epoch:
                utils.save_chkpt(model, optim, checkpoints_dir, epoch=epoch)

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = model_input.cuda()
                gt = gt.cuda()
                start_time = time.time()

                restored = model(model_input)
                train_loss = loss_fn(restored, gt)

                optim.zero_grad(set_to_none=True)
                train_loss.backward()
                optim.step()
                pbar.update(1)
                if not total_steps % args.steps_til_ckpt:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                if not total_steps % args.steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    summary_fn(gt, restored, total_steps, optim)
                    tqdm.write("Epoch %d, Total loss %0.6f, "
                               "iteration time %0.6f, total time %0.6f" % (
                                   epoch, train_loss, time.time() - start_time,
                                   time.time() - total_time_start))

                    if val_dataloader is not None:
                        with torch.no_grad():
                            model.eval()

                            val_losses = []
                            val_psnrs = []
                            val_lpips = []

                            val_ssims = []
                            for (model_input, gt) in tqdm(val_dataloader):
                                model_input = model_input.cuda()
                                gt = gt.cuda()
                                restored = model(model_input)
                                val_loss = loss_fn(restored, gt)
                                psnr = summary_utils.get_psnr(restored, gt)
                                ssim = summary_utils.get_ssim(restored, gt)
                                vlpips = custom_loss_fn(restored, gt)
                                val_ssims.append(ssim)
                                val_psnrs.append(psnr)
                                val_losses.append(val_loss.item())
                                val_lpips.append(vlpips)

                            summary_utils.write_val_scalars(writer, ['psnr', 'ssim', 'lpips', 'loss'],
                                                            [val_psnrs, val_ssims, val_lpips, val_losses], total_steps)
                            all_val_psnrs.append(np.mean(val_psnrs))

                            if np.mean(val_psnrs) > best_val_psnr:
                                print(f'BEST PSNR: {np.mean(val_psnrs)}')
                                best_val_psnr = np.mean(val_psnrs)
                                torch.save(model.state_dict(),
                                           os.path.join(checkpoints_dir, 'model_best.pth'))
                                utils.save_best(args, total_steps, epoch, val_psnrs, val_ssims, val_lpips, checkpoints_dir)

                        model.train()
                        if args.sched == 'reducelr':
                            scheduler.step(val_loss)

                    if args.sched == 'cosine':
                        scheduler.step()
                    elif args.sched == 'warm':
                        scheduler.step(epoch + step / args.s_iters)
                    elif args.sched == 'multi':
                        scheduler.step(total_steps)

                total_steps += 1
        utils.save_chkpt(model, optim, checkpoints_dir, epoch=epoch, val_psnrs=all_val_psnrs, final=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/custom_data')
    parser.add_argument('--log_root',
                        type=str,
                        default='/home/cindy/PycharmProjects/img-restoration-zoo/logs')
    parser.add_argument('--test', action='store_true',
                        help='dummy experiment name for just testing code, use to test pipes')
    parser.add_argument('-ds', '--dataset', help='dataset', type=str, choices=['gopro', 'nfs'], default='nfs')
    parser.add_argument('-b', '--block_size',
                        help='delimited list input for block size',
                        default='8, 512, 512')
    parser.add_argument('--resume',
                        type=str,
                        default='00-00',
                        help='date of folder of exp to resume')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--sched', type=str, help='schedule', default='no_sched')

    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--lr', help='model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--steps_til_summary', type=int, default=2000)
    parser.add_argument('--epochs_til_checkpoint', type=int, default=200)
    parser.add_argument('--steps_til_ckpt', type=int, default=3000)
    parser.add_argument('--no_cudnn', action='store_true')

    parser.add_argument('--loss', type=str, choices=['def', 'l1', 'l2'], default='def')
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'mpr', 'hi', 'dncnn', 'sgn', 'uform', 'drn', 'banet', 'msrn'],
                        default='unet')

    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(',')]
    main(args)
