import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataprep.data_utils import *

folders = {'gopro': 'GOPRO_Large_all',
           'nfs': 'nfs240_local'}

REMOTE_DATA_PATH = '/media/data6/cindy/data'
LOCAL_DATA_PATH = '/home/cindy/PycharmProjects/data'

REMOTE_CUSTOM_PATH = '/media/data6/cindy/custom_data'
LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'


def gopro_init(log_root, block_size, data_root, split, num_blocks_per_video):
    log_root = f'{log_root}_{block_size[0]}f'
    # Create output directory for prepared dataset
    os.makedirs(f'{log_root}/{split}', exist_ok=True)

    save_dir = f'{log_root}/{split}'

    save_dict = {}

    dataset = 'gopro'
    data_root = f'{data_root}/{folders[dataset]}'
    # Create video blocks
    subfolder_num = 0
    for subfolder_path in sorted(glob.glob(f'{data_root}/{split}/*', recursive=True)):
        # create a list of the frames of length N of arrays [C, H, W]
        print(f'Gathering images in {subfolder_path}')
        video_seq = convert_gopro_frames(subfolder_path, ext='/*.png', num_blocks_per_video=num_blocks_per_video,)

        # normalize [0,1], convert uint8->float32, create a video array C H W
        video = convert_gopro_seq(video_seq, block_size, crop=True)
        print(f'Created video for {subfolder_num}, size={video.shape}')
        # create and save video blocks
        vid_dict = save_vid_dict(video, num_items_per_video=num_blocks_per_video)
        save_dict[subfolder_num] = vid_dict
        subfolder_num += 1
    torch.save(save_dict, f'{save_dir}/blocks_per_vid{num_blocks_per_video}.pt')


def nfs_annotate(log_root, block_size, data_root, split):
    log_root = f'{log_root}_{block_size[0]}f'
    files = [f for f in glob.glob(f'{data_root}/*') if 'MACOS' not in f]
    save_dir = f'{log_root}/{split}'
    os.makedirs(save_dir, exist_ok=True)

    for i, fpath in enumerate(files):
        subject = fpath.split('/')[-1]
        subfolder_path = f'{fpath}/240/{subject}'

        print(subfolder_path)
        video_seq, num_vids, start_idxs = create_nfs_seq(subfolder_path, num_vids=20)
        nfs_annotate(video_seq, block_size, subject=subject, start_idxs=start_idxs)


def nfs_init(log_root, block_size, data_root, split):
    log_root = f'{log_root}_{block_size[0]}f'
    subject_folders = [f for f in glob.glob(f'{data_root}/nfs240_local/*') if 'MACOS' not in f]
    assert len(subject_folders) != 0

    matchers = ['kid_swing', 'biker_head_3']
    matching = [s for s in subject_folders if any(xs in s for xs in matchers)]

    if split == 'train':
        for i, thing in enumerate(matching):
            if thing in subject_folders:
                subject_folders.remove(thing)
    else:
        subject_folders = matching

    save_dir = f'{log_root}/{split}'

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv('nfs_crop_idx.csv')
    save_dict = {}
    subfolder_num = 0
    for i, fpath in enumerate(subject_folders):

        subject = fpath.split('/')[-1]
        found = df[df['subject'] == subject].count()['subject']
        if found:
            print(subject)
            subfolder_path = f'{fpath}/240/{subject}'
            sub_df = df[df['subject'] == subject]
            num_vids = len(sub_df)
            print(subfolder_path)

            video_seq = create_nfs_seq_from_annotation(subfolder_path, block_size, sub_df)
            video = convert_nfs_seq(video_seq, sub_df, block_size, save_dir, subject=subject)

            print(f'Created video for {subfolder_num}, size={video.shape}')
            vid_dict = save_vid_dict(video, num_items_per_video=num_vids, num_frames=block_size[0])
            save_dict[subfolder_num] = vid_dict
            subfolder_num += 1
    torch.save(save_dict, f'{save_dir}/filter.pt')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-l', '--list', help='delimited list input', type=str, default='8,512,512')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['nfs', 'gopro'], default='nfs')

    args = parser.parse_args()

    log_root = f'{REMOTE_CUSTOM_PATH}'
    block_size = [int(item) for item in args.list.split(',')]


    if args.remote:
        log_root = f'{REMOTE_CUSTOM_PATH}/{args.dataset}_block_rgb_{block_size[-1]}'
        data_root = REMOTE_DATA_PATH
    else:
        log_root = f'{LOCAL_CUSTOM_PATH}/{args.dataset}_block_rgb_{block_size[-1]}'
        data_root = LOCAL_DATA_PATH

    if args.dataset == 'gopro':
        gopro_init(log_root=log_root,
                   block_size=block_size,
                   data_root=data_root,
                   split='train',
                   num_blocks_per_video=30)
        gopro_init(log_root=log_root,
                   block_size=block_size,
                   data_root=data_root,
                   split='test',
                   num_blocks_per_video=5)
    elif args.dataset == 'nfs':

        ''' Uncomment to annotate and select where to crop frames'''
        # nfs_annotate(log_root, block_size, data_root, split)
        nfs_init(log_root=log_root,
                 block_size=block_size,
                 data_root=data_root,
                 split='train')
        nfs_init(log_root=log_root,
                   block_size=block_size,
                   data_root=data_root,
                   split='test')


