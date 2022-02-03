import os
import random

import numpy as np
import torch.utils.data as torch_data
from PIL import Image

import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train', subsample=-1, shuffle_subsample=None, self_training='label_2', self_training_textfile=None):
        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        if subsample > 0 and split == 'train':
            if shuffle_subsample is not None:
                split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', 'train_car1_{}.txt'.format(shuffle_subsample))
                if not os.path.isfile(split_dir):
                    temp = os.path.join(root_dir, 'KITTI', 'ImageSets', 'train_car1.txt')
                    temp = [x.strip() for x in open(temp).readlines()]
                    random.shuffle(temp)
                    with open(split_dir, 'w') as f:
                        for item in temp:
                            f.write('{}\n'.format(item))
            else:
                split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', 'train_car1.txt'.format(shuffle_subsample))
            
            if self_training != 'label_2':
                if shuffle_subsample is not None:
                    temp = os.path.join(root_dir, 'KITTI', 'ImageSets', self_training_textfile)
                    temp = [x.strip() for x in open(temp).readlines()]
                    random.shuffle(temp)
                    self.image_idx_list = temp[:subsample]
                    #self.image_idx_list = random.shuffle([x.strip() for x in open(os.path.join(root_dir, 'KITTI', 'ImageSets', self_training_textfile)).readlines()])[:subsample]
                else:
                    self.image_idx_list = [x.strip() for x in open(os.path.join(root_dir, 'KITTI', 'ImageSets', self_training_textfile)).readlines()][:subsample]
            else:
                self.image_idx_list = [x.strip() for x in open(split_dir).readlines()][:subsample]
        else:
            if self_training != 'label_2':
                self.image_idx_list = [x.strip() for x in open(os.path.join(root_dir, 'KITTI', 'ImageSets', self_training_textfile)).readlines()]
            else:
                split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
                self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        #for self training
        if self_training != 'label_2':
            self.label_dir = os.path.join(self.imageset_dir, self_training) #--self_training needs to describe the label folder with the self-training labels
        else:
            self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
