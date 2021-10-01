import sys, os
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataio import GoPro_Video, NFS_Video

REMOTE_GOPRO_PATH = '/media/data6/cindy/data/GOPRO_Large_all'
LOCAL_GOPRO_PATH = '/home/cindy/PycharmProjects/data/GOPRO_Large_all'

REMOTE_CUSTOM_PATH = '/media/data6/cindy/custom_data'
LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'


def loadTrainingDataset(args, color=False, input_avg=False):
    if args.remote:
        args.data_root = f'{REMOTE_CUSTOM_PATH}/{args.dataset}_block_rgb_{args.block_size[-1]}_8f'
    else:
        args.data_root = f'{LOCAL_CUSTOM_PATH}/{args.dataset}_block_rgb_{args.block_size[-1]}_8f'
    split = 'train'

    if args.dataset == 'gopro':
        train_dataset = GoPro_Video(log_root=args.data_root,
                                    block_size=args.block_size,
                                    gt_index=args.gt,
                                    input_avg=input_avg,
                                    color=color,
                                    split=split)
    else:
        train_dataset = NFS_Video(log_root=args.data_root,
                                            block_size=args.block_size,
                                            gt_index=args.gt,
                                            input_avg=input_avg,
                                            color=color,
                                            split=split)

    return DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=True,
                      pin_memory=True)


def loadValDataset(args, input_avg=False, color=False):
    if args.remote:
        args.data_root = f'{REMOTE_CUSTOM_PATH}/{args.dataset}_block_rgb_{args.block_size[-1]}_8f'
    else:
        args.data_root = f'{LOCAL_CUSTOM_PATH}/{args.dataset}_block_rgb_{args.block_size[-1]}_8f'

    split = 'test'

    if args.dataset == 'gopro':
        val_dataset = GoPro_Video(log_root=args.data_root,
                                  gt_index=args.gt,
                                  block_size=args.block_size,
                                  input_avg=input_avg,
                                  split=split,
                                  color=color)
    else:
        val_dataset = NFS_Video(log_root=args.data_root,
                        block_size=args.block_size,
                        gt_index=args.gt,
                        input_avg=input_avg,
                        color=color,
                        split=split)

    return DataLoader(val_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=False,
                      pin_memory=True)
