'''
Test script will save results into a csv file
'''
import argparse
import lpips
import os
import sys
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim
from tqdm.autonotebook import tqdm

# Enable import from parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
from utils import find_version_number, make_subdirs, save_img
import dataloading
import utils
import summary_utils
import nets.loss

utils.seed(1234)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--folder_name', type=str, default='21-09-30')
    args = parser.parse_args()
    log_root = f'/home/cindy/PycharmProjects/img-restoration-zoo/logs/{args.folder_name}'

    if os.path.exists(f'{log_root}/idv_results.csv'):
        os.remove(f'{log_root}/idv_results.csv')
    if os.path.exists(f'{log_root}/mean_results.csv'):
        os.remove(f'{log_root}/mean_results.csv')

    folders = sorted([f for f in os.listdir(log_root) if '.csv' not in f])

    mean_column_names = ['exp',
                         'decoder',
                         'lr',
                         'schedule',
                         'LPIPS',
                         'PSNR',
                         'SSIM']

    loss_fn = lpips.LPIPS(net='vgg').cuda()
    custom_loss_fn = nets.loss.lpips_1d
    l1_loss = nn.L1Loss()

    # put indices of images you would like to save for viewing
    save_imgs = [0, 1]

    for i in range(len(save_imgs)):
        mean_column_names.append(f'PSNR_{save_imgs[i]}')

    for j, date_model in enumerate(folders):

        sub_root = f'{log_root}/{date_model}'
        experiments = sorted([f for f in os.listdir(sub_root) if '.csv' not in f])

        for i, exp in enumerate(experiments):
            exp_root = f'{sub_root}/{exp}'
            version_num = find_version_number(exp_root)
            model_dir = f'{exp_root}/v_{version_num - 1}'  # get the latest version
            print(model_dir)

            with open(f'{exp_root}/args.json', 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(namespace=t_args)
            args.remote = False

            summaries_dir, checkpoints_dir = make_subdirs(model_dir, make_dirs=False)

            model = models.define_decoder(args.decoder, args)
            model.cuda()
            model.eval()

            model.load_state_dict(torch.load(f'{model_dir}/checkpoints/model_best.pth'))

            os.makedirs(f'{exp_root}/deblurred', exist_ok=True)
            os.makedirs(f'{exp_root}/gt', exist_ok=True)

            dfs = []

            val_dataloader = dataloading.loadValDataset(args)

            loss_postnet = []
            psnr_postnet = []
            ssim_postnet = []

            psnrs = []

            count = 0

            full_time = 0.0

            with torch.no_grad():

                for (model_input, gt) in tqdm(val_dataloader):
                    model_input = model_input.cuda()
                    gt = gt.cuda()
                    restored = model(model_input)

                    #### pre
                    loss = custom_loss_fn(loss_fn, restored.squeeze(0), gt.squeeze(0))
                    loss_postnet.append(loss)
                    p = summary_utils.get_psnr(restored, gt)
                    s = summary_utils.get_ssim(restored, gt)

                    psnr_postnet.append(p)
                    ssim_postnet.append(s)

                    if count in save_imgs:
                        disp = True
                        save_img(restored, count, f'{exp_root}/deblurred', display=disp)
                        save_img(gt, count, f'{exp_root}/gt', display=disp)
                        psnrs.append(round(p, 3))

                    single_column_names = ['Model', 'Decoder', 'PSNR', 'SSIM', 'LPIPS']

                    df = pd.DataFrame(columns=single_column_names)
                    series = pd.Series([exp_root,
                                        args.decoder,
                                        round(p, 3),
                                        round(s, 3),
                                        round(loss, 3)],
                                       index=df.columns)
                    df = df.append(series, ignore_index=True)

                    file_name = f'{log_root}/idv_results.csv'
                    # if file does not exist write header
                    if not os.path.isfile(file_name):
                        df.to_csv(file_name, header=single_column_names)
                    else:  # else it exists so append without writing the header
                        df.to_csv(file_name, mode='a', header=False)

                    count += 1

            df = pd.DataFrame(columns=mean_column_names)
            results = [exp,
                       args.decoder,
                       args.lr,
                       args.sched,
                       np.round(np.mean(loss_postnet), decimals=3),
                       np.round(np.mean(psnr_postnet), decimals=3),
                       np.round(np.mean(ssim_postnet), decimals=3)]
            for psnr in psnrs:
                results.append(psnr)


            series = pd.Series(results, index=df.columns)
            df = df.append(series, ignore_index=True)

            file_name = f'{log_root}/mean_results.csv'
            # if file does not exist write header
            if not os.path.isfile(file_name):
                df.to_csv(file_name, header=mean_column_names)
            else:  # else it exists so append without writing the header
                df.to_csv(file_name, mode='a', header=False)

    sns.set_style('whitegrid')

    '''
    Following code can be used to visualize some x-axis across 3 metrics.
    Must log this additional value as a column in idv_results.csv
    '''
    # data = pd.read_csv(f'{log_root}/idv_results.csv')
    #
    # f, axes = plt.subplots(1, 3)
    # sns.lineplot(data=data, x="Shot Noise Stdv", y="PSNR", hue="Model", ax=axes[0])
    # axes[0].set_title('PSNR')
    # sns.lineplot(data=data, x="Shot Noise Stdv", y="SSIM", hue="Model", ax=axes[1])
    # axes[1].set_title('SSIM')
    # sns.lineplot(data=data, x="Shot Noise Stdv", y="LPIPS", hue="Model", ax=axes[2])
    # axes[2].set_title('LPIPS')
    # plt.show()
