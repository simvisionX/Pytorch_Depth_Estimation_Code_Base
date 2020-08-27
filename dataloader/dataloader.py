from __future__ import absolute_import, division, print_function

from torch.utils.data import Dataset
import os

from utils import utils
from utils.file_io import read_img, read_disp


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.load_pseudo_gt = load_pseudo_gt
        self.transform = transform

        sceneflow_finalpass_dict = {
            'train': 'filenames/SceneFlow_finalpass_train.txt',
            'val': 'filenames/SceneFlow_finalpass_val.txt',
            'test': 'filenames/SceneFlow_finalpass_test.txt'
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        dataset_name_dict = {
            'SceneFLow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_MIX': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]
        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()
            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]

            sample_path = {}

            if self.save_filename:
                sample_path['left_name'] = left_img.split('/', 1)[1]

            sample_path['left'] = os.path.join(self.data_dir, left_img)
            sample_path['right'] = os.path.join(self.data_dir, right_img)
            sample_path['disp'] = os.path.join(self.data_dir, gt_disp) if gt_disp is not None else None

            if self.load_pseudo_gt and sample_path['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample_path['disp']:
                    sample_path['pseudo_disp'] = sample_path['disp'].replace('disp_occ_0',
                                                                   'disp_occ_0_pseudo_gt')
                #KITTI 2012
                elif 'disp_occ' in sample_path['disp']:
                    sample_path['pseudo_disp'] = sample_path['disp'].replace('disp_occ',
                                                                   'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample_path['pseudo_disp'] = None

            self.samples.append(sample_path)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path)['left']
        sample['right'] = read_img(sample_path)['right']

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        # TODO: 这里说是要判断subset，但是看了一圈好像没有这个词， 并且前面assert说dataset_name必须要在一个字典里，也没有subset
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)











