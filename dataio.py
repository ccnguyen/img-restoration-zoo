from utils import *
from torch.utils.data import Dataset


def torch_rgb2gray(vid):
    vid[:, 0, :, :] *= 0.2989
    vid[:, 1, :, :] *= 0.5870
    vid[:, 2, :, :] *= 0.1140
    return torch.sum(vid, dim=1)


class GoPro_Video(Dataset):
    def __init__(self,
                 log_root='/home/cindy/PycharmProjects/custom_data/gopro_block_rgb_512_8f',
                 block_size=[8, 512, 512],
                 gt_index=4,
                 split='train',
                 color=False):
        super().__init__()

        self.log_root = log_root
        self.block_size = block_size
        self.split = split
        self.gt_index = gt_index
        self.color = color

        # load video block names
        print('creating list of video blocks')
        self.video_blocks = []
        if self.split == 'test' or self.split == 'val':
            fname = 'blocks_per_vid5.pt'
        else:
            fname = 'blocks_per_vid30.pt'

        fpath = f'{self.log_root}/{self.split}/{fname}'

        self.vid_dict = torch.load(fpath)

        self.num_vids = max(self.vid_dict.keys()) + 1
        self.num_clips_each = max(self.vid_dict[0].keys()) + 1

        self.num_clips_total = self.num_vids * self.num_clips_each
        print(f'loaded {self.num_clips_total} clips from {self.num_vids} videos')

    def __len__(self):
        return self.num_clips_total

    def __getitem__(self, idx):
        vid_num = idx // self.num_clips_each
        clip_num = idx % self.num_clips_each

        vid = self.vid_dict[vid_num][clip_num]  # 8, 3, H, W

        if self.color:
            gt = vid[self.gt_index, ...]
            avg = torch.mean(vid, dim=0)
        else:
            vid = torch_rgb2gray(vid.clone())
            avg = torch.mean(vid, dim=0, keepdim=True) # [1, H, W]

            gt = vid[self.gt_index, ...]
            gt = gt.unsqueeze(0) # [1, H, W]
            if self.split != 'test' and self.split != 'sample':
                [avg, vid, gt] = augmentData([avg, vid, gt])

            # avg [1,h,w]
            # vid [8, h,w], you can pass a video too
            # gt  [1, h,w]
        return avg, gt



class NFS_Video(Dataset):
    def __init__(self,
                 log_root='/home/cindy/PycharmProjects/custom_data/nfs_block_rgb_512_8f',
                 block_size=[8, 512, 512],
                 gt_index=4,
                 split='train',
                 color=False):
        '''
        3 x 1280 x 720 pixels originally
        init blocks will make it 512 x 512
        '''
        super().__init__()

        self.log_root = log_root
        self.block_size = block_size
        self.split = split
        self.gt_index = gt_index
        self.color = color

        # load video block names
        print('creating list of video blocks')
        self.video_blocks = []
        fname = 'filter.pt'

        fpath = f'{self.log_root}/{self.split}/{fname}'
        if self.split == 'sample':
            raise NotImplementedError('no sample dataset for nfs')

        self.vid_dict = torch.load(fpath)

        self.num_vids = max(self.vid_dict.keys()) + 1

        self.num_clips_each = []
        for i in range(self.num_vids):
            vid = self.vid_dict[i]
            self.num_clips_each.append(max(vid.keys()) + 1)

        self.num_clips_total = sum(self.num_clips_each)
        # self.num_clips_total = self.num_vids * self.num_clips_each
        print(f'loaded {self.num_clips_total} clips from {self.num_vids} videos')

        self.stop_idx = np.cumsum(np.array(self.num_clips_each))

        vid_mapping = {}

        vid_num = 0
        clip_num = 0
        for i in range(self.num_clips_total):
            if i == self.stop_idx[vid_num]:
                vid_num += 1
                clip_num = 0
            vid_mapping[i] = (vid_num, clip_num)
            clip_num += 1
        self.vid_mapping = vid_mapping

    def __len__(self):
        return self.num_clips_total

    def __getitem__(self, idx):
        (vid_num, clip_num) = self.vid_mapping[idx]
        vid = self.vid_dict[vid_num][clip_num]  # 8, 3, H, W

        if self.color:
            gt = vid[self.gt_index, ...]
            avg = torch.mean(vid, dim=0)
        else:
            vid = torch_rgb2gray(vid.clone())
            avg = torch.mean(vid, dim=0, keepdim=True) # [1, H, W]

            gt = vid[self.gt_index, ...]
            gt = gt.unsqueeze(0) # [1, H, W]
            if self.split != 'test' and self.split != 'sample':
                [avg, vid, gt] = augmentData([avg, vid, gt])

            # avg [1,h,w]
            # vid [8, h,w]
            # gt  [1, h,w]
        return avg, gt
