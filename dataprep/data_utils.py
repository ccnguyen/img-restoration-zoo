import torch
import glob
import skimage.io
from tqdm import tqdm
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

random.seed(1234)

def crop_center(img, size_x, size_y):
    _, y, x = img.shape
    startx = x // 2 - (size_x // 2)
    starty = y // 2 - (size_y // 2)
    return img[..., starty:starty + size_y, startx:startx + size_x]


def crop_idx(img, y, x):
    c, h, w = img.shape
    if h < y + 512:
        y = h - 512
    if w < x + 512:
        x = w - 512
    return img[..., y:y+512, x:x+512]


def load_gopro_frames(fpath, block_size, num_blocks_per_vid=8):
    all_frames = sorted(os.listdir(fpath))
    num_frames = len(all_frames)

    video_seq = []
    for i in range(num_blocks_per_vid):
        start_idx = random.randint(0, num_frames - block_size[0])
        for j in range(block_size[0]):
            img = skimage.io.imread(f'{fpath}/{all_frames[start_idx+j]}')
            video_seq.append(img)
    return video_seq


def convert_gopro_frames(folder_path, ext, num_blocks_per_video=100, num_frames=8, grayscale=True):
    folders = sorted(glob.glob(folder_path + ext, recursive=True))
    video_seq = []
    for i, fname in enumerate(folders):
        if i % 10 < num_frames:
            # skip some frames in between videos to get unique videos
            # e.g. for 8 frame blocks, it skips the next 2 frames
            if grayscale:
                im = skimage.io.imread(fname, as_gray=True)  # H, W
            else:
                im = skimage.io.imread(fname)  # H, W, 3

            video_seq.append(im)  # append each frame to a list
        if i == num_blocks_per_video * 10 - 1:
            break # stop video once we get enough frames
    print(f'number of frames: {len(video_seq)}')
    return video_seq  # list length N, each item is [H, W, C=3]



def create_nfs_seq(fpath, block_size, num_vids=None):
    all_frames = sorted(os.listdir(fpath))
    num_frames_to_use = np.minimum(len(all_frames), 3000)
    num_avail_frames = len(all_frames)

    if num_vids is None:
        num_vids = num_frames_to_use // block_size[0]
    start_idxs = []

    video_seq = []
    for i in range(num_vids):
        # random determine where to start the video
        start_idx = random.randint(0, num_avail_frames - block_size[0])
        start_idxs.append(start_idx)
        for j in range(block_size[0]):
            img = skimage.io.imread(f'{fpath}/{all_frames[start_idx + j]}')
            video_seq.append(img)
    return video_seq, num_vids, start_idxs


def create_nfs_seq_from_annotation(fpath, block_size, df):
    all_frames = sorted(os.listdir(fpath))
    num_vids = len(df)

    video_seq = []
    for i in range(num_vids):
        start_idx = df['start_idx'].iloc[i]
        for j in range(block_size[0]):
            img = skimage.io.imread(f'{fpath}/{all_frames[start_idx + j]}')
            video_seq.append(img)
    return video_seq


def annotate_nfs(video_seq, subject=None, start_idxs=None):
    # data conversion and create an array containing all the frames
    assert len(video_seq) > 0, 'video seq must contain at least one frame'
    N = len(video_seq)

    header = ['subject', 'start_idx', 'start_y', 'start_x']

    clip_count = 0
    for i in range(N):
        im = torch.from_numpy(video_seq[i]).type(torch.float32)
        im = im.permute(2, 0, 1)  # permute from numpy format to torch format
        im = im / 255.0  # [0,1]
        if i % 8 == 0:
            plt.figure()
            plt.imshow(np.transpose(im, (1, 2, 0)))
            plt.grid()

            plt.show()
            start_y = int(input("start_y: "))
            start_x = int(input("start_x: "))
            print(f'clip: {clip_count}')

            df = pd.DataFrame(columns=header)
            series = pd.Series([subject, start_idxs[clip_count], start_y, start_x])
            df = df.append(series, ignore_index=True)
            file_name = f'nfs_crop_idx.csv'
            # if file does not exist write header
            if not os.path.isfile(file_name):
                df.to_csv(file_name, columns=header)
            else:  # else it exists so append without writing the header
                df.to_csv(file_name, mode='a', header=False)
            clip_count += 1
    return None


def convert_gopro_seq(video_seq, block_size=[8, 512, 512], save_dir='', crop=False, subject=None, start_idxs=None):
    # data conversion and create an array containing all the frames
    assert len(video_seq) > 0, 'video seq must contain at least one frame'
    N = len(video_seq)
    if crop:
        C, H, W = 3, block_size[-2], block_size[-1]
    else:
        H, W, C = video_seq[0].shape
    video = torch.zeros([N, C, H, W], dtype=torch.float32)

    for i in range(N):
        im = torch.from_numpy(video_seq[i]).type(torch.float32)
        im = im.permute(2, 0, 1)  # permute from numpy format to torch format
        im = im / 255.0  # [0,1]
        if crop:
            im = crop_center(im, 512, 512)

        video[i, :, :, :] = im
    return video


def convert_nfs_seq(video_seq, df, block_size=[8, 512, 512], save_dir='', subject=None):
    # data conversion and create an array containing all the frames
    assert len(video_seq) > 0, 'video seq must contain at least one frame'
    N = len(video_seq)
    C, H, W = 3, block_size[-2], block_size[-1]

    num_vids = len(df)
    video = torch.zeros([N, C, H, W], dtype=torch.float32)

    frame_count = 0
    for i in range(num_vids):
        start_y = df['start_y'].iloc[i]
        start_x = df['start_x'].iloc[i]
        for j in range(block_size[0]):
            im = torch.from_numpy(video_seq[i * 8 + j]).type(torch.float32)

            im = im.permute(2, 0, 1)  # permute from numpy format to torch format
            im = im / 255.0  # [0,1]
            im = crop_idx(im, start_y, start_x)
            video[frame_count, :, :, :] = im
            frame_count += 1
    return video


def imsave(fname, x, dataformats='CDHW'):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if x.ndim == 4:
        if dataformats == 'CDHW':
            x = x.transpose(1, 2, 3, 0)

    if x.ndim == 3:
        x = x.transpose(1, 2, 0)
    skimage.io.imsave(fname, x, check_contrast=False)


def save_vid_dict(video, num_items_per_video=200, num_frames=8):
    vid_dict = {}
    for idx in tqdm(range(num_items_per_video)):
        # chop up block of frames to their idv 8 frame videos
        clip = video[idx * num_frames:idx * num_frames + num_frames, ...].clone()
        vid_dict[idx] = clip
    return vid_dict