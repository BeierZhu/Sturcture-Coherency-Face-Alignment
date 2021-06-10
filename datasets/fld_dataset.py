# Custom "Dataset" class for LABA
# Author: Beier ZHU
# Date: 2019/07/25

import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib
matplotlib.use('agg')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2 
from datasets.align_transforms import RandomAffine, CenterCrop, Grayscale, RandomOcclude, \
                                      ToTensor, Normalize, GuidedAffine, _get_corr_list, RandomBlur, \
                                      ToBoundaryHeatmap
from torchvision import transforms
from utils.log_helper import cprint
from utils.misc import load_mean_pose
import copy

cv2.ocl.setUseOpenCL(False)

class TransformBuilder(object):
    def __init__(self, config):
        self.config = config

    def train_transforms(self):
        config = self.config
        cor_list = _get_corr_list(config.structure.num_kpts)

        train_transforms = []
        train_transforms.append(RandomAffine(degrees=config.train_param.rotation, 
                                             translate=(config.train_param.translate_rate, config.train_param.translate_rate),
                                             scale=(config.train_param.scale_min, config.train_param.scale_max), 
                                             mirror=config.train_param.mirror, corr_list=cor_list))
        train_transforms.append(CenterCrop(config.structure.crop_size))
        if hasattr(config.train_param, 'is_occluded') and config.train_param.is_occluded:
            cprint.yellow('Add Occlusion To Images ......')
            train_transforms.append(RandomOcclude(config.structure.crop_size))
        if hasattr(config.structure, 'is_color') and not config.structure.is_color:
            cprint.yellow('Feed Grayscale Images ......')
            train_transforms.append(Grayscale())
        if hasattr(config.train_param, 'is_blurred') and config.train_param.is_blurred:
            cprint.yellow('Blur Images ......')
            train_transforms.append(RandomBlur())
        if hasattr(config.train_param, 'use_boundary') and config.train_param.use_boundary:
            cprint.yellow('Generat Boundary Heatmap ......')
            train_transforms.append(ToBoundaryHeatmap())
        train_transforms.append(Normalize())
        train_transforms.append(ToTensor())
            
        return transforms.Compose(train_transforms)

    def test_transforms(self):
        config = self.config

        test_transforms = []
        test_transforms.append(RandomAffine(degrees=0, translate=None,
                        scale=((config.train_param.scale_min + config.train_param.scale_max) / 2, \
                        (config.train_param.scale_min + config.train_param.scale_max) / 2),
                        mirror=None, corr_list=None))
        test_transforms.append(CenterCrop(config.structure.crop_size))
        if hasattr(config.structure, 'is_color') and not config.structure.is_color:
            cprint.yellow('Feed Grayscale Images ......')
            test_transforms.append(Grayscale())
        test_transforms.append(Normalize())
        test_transforms.append(ToTensor())

        return transforms.Compose(test_transforms)

    def fine_transforms(self):
        config = self.config
        mean_pose = load_mean_pose(config.test_param.mean_pose_path)

        fine_transforms = []
        fine_transforms.append(GuidedAffine(guided_pose=mean_pose))
        fine_transforms.append(CenterCrop(config.structure.crop_size))
        if not config.structure.is_color:
            cprint.yellow('Feed Grayscale')
            fine_transforms.append(Grayscale())
        fine_transforms.append(Normalize())
        fine_transforms.append(ToTensor())

        return transforms.Compose(fine_transforms)

class FLDDS(data.Dataset):
    """ FLDDS: Facial Landmark Detector Data Set.

    Args:
        root_folders (list of string): List of root folders where images are located
        sources (list of string): List of paths to txt annotation file
        transform (callable, optional): A function/transform that takes in an PIL images and labels,
             and returns a transformed version
        num_points (int, optional): #face landmarks
    """
    def __init__(self, root_folders, sources, transform=None, num_points=68, is_pdb=False, num_bins=9):
        self.root_list = root_folders.split(' ')
        self.source_list = sources.split(' ')
        
        assert len(self.root_list) == len(self.source_list),\
            '#image folders should equal to #annotation files, but got %d and %d' \
            %(len(self.root_list), len(self.source_list))

        self.transform = transform
        self.num_points = num_points

        assert self.num_points in [29, 68, 98, 106],\
            '#landmarks number expected to be {29, 68, 98, 106}, got %d' %self.num_points
        # self.num_points*2 + 1. w/o weights || self.num_points*2*2 + 1. with weights
        valid_anno_num = [self.num_points * 2 + 1, self.num_points * 2 * 2 + 1]

        self.metas = []
        self.landmarks = [] # Only for PDB strategy

        for root_folder, source in zip(self.root_list, self.source_list):
            print("Building input data from annotation file: ", source)
            with open(source) as f:
                lines = f.readlines()

            for line in lines:
                if line in ['', '\n']: continue

                elements = line.strip('\n').strip(' ').split(' ')
                assert len(elements) in valid_anno_num,\
                    '#annotation w.r.t %d landmarks should be %d or %d, but got %d' \
                    %(self.num_points, valid_anno_num[0], valid_anno_num[1], len(elements))
                img_name = elements[-1]
                img_path = os.path.join(root_folder, elements[-1])
                points = [float(element) for element in elements[:self.num_points*2]]
                points = np.array(points)
                self.landmarks.append(points)
                self.metas.append((img_name, img_path, points))

        self.num = len(self.metas)
        print("Data Loading in FLDDS with size: %d" %self.num)

        self.landmarks = np.stack(self.landmarks)

        if is_pdb:
            pdb_metas = []
            sample_weights = self._cal_PDB_weights(num_bins=num_bins)
            for i, repeat in enumerate(sample_weights):
                for _ in range(repeat):
                    pdb_metas.append(copy.deepcopy(self.metas[i]))

            self.metas = pdb_metas
            self.num = len(self.metas)
            print("Data Loading with PDB strategy size : {}".format(self.num))


    @staticmethod
    def _map_to_idx(value, bins):
        for i in range(1, len(bins) - 1):
            if value < bins[i]:
                return i - 1

        return len(bins) - 2

    def _cal_PDB_weights(self, num_bins=9):
        pca = PCA(n_components=1)
        projected = pca.fit_transform(self.landmarks)[:, 0]
        projected = np.abs(projected)
        n, bins, _ = plt.hist(projected, bins=num_bins)
        prob = np.round(max(n)/n).astype(int) 

        sample_weights = []
        for p in projected:
            sample_weight = prob[FLDDS._map_to_idx(p, bins)]
            sample_weights.append(sample_weight)

        return sample_weights

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        img_name, img_path, points = self.metas[idx]
        img = cv2.imread(img_path)

        # attribute 'name', 'image_origin' and 'landmarks_origin' 
        # remain unchanged during self.transform 
        sample = {'name': img_name, 
                'image_origin': img.copy(), 'landmarks_origin': points.copy(),
                'image': img.copy(), 'landmarks': points.copy()}

        # transform
        if self.transform is not None:
            sample = self.transform(sample)

        sample['image'] = torch.FloatTensor(sample['image'])
        sample['landmarks'] = torch.FloatTensor(sample['landmarks'])
        
        if 'boundary' in sample.keys():
            sample['boundary'] = torch.FloatTensor(sample['boundary'])
        if 'heatmap' in sample.keys():
            sample['heatmap'] = torch.FloatTensor(sample['heatmap'])

        return sample

