import os
import random

import torch
import numpy as np
import torch.utils.data as data
from torchvision.transforms import Normalize, ToTensor
from PIL import Image

from .io import IO
from .build import DATASETS
from utils.logger import *


cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16,
'02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33,
'03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50,
'04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class ShapeNetImagePoint(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class ShapeNetRender(ShapeNet):
    def __init__(self, config):
        # partition='train', whole=False, num_points=1024):
        super().__init__(config)
        # self.partition = config.subset
        self.views_dist = torch.ones((10), dtype=torch.float, requires_grad=False)
        self.views_elev = torch.asarray((0, 90, 180, 270, 225, 225, 315, 315, 0, 0), dtype=torch.float,
                                        requires_grad=False)
        self.views_azim = torch.asarray((0, 0, 0, 0, -45, 45, -45, 45, -90, 90), dtype=torch.float, requires_grad=False)
        # self.render = Renderer()
        self.totensor = ToTensor()
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        points = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        points = self.random_sample(points, self.sample_points_num)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()


        name = sample['taxonomy_id'] + '_' + sample['model_id']
        rand_idx = random.randint(0, 9)
        image = Image.open(os.path.join(self.data_root,'rendering/%s/%d.png' % (name, rand_idx)))
        image = self.norm(self.totensor(image))
        return image, points, self.views_azim[rand_idx], self.views_elev[rand_idx], self.views_dist[rand_idx]