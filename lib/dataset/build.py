# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import BeanDataset, OutputDataset
from .COCODataset import CocoRescoreDataset as rescore_coco
from .COCOKeypoints import BeanKeypoints
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseDataset import CrowdPoseRescoreDataset as rescore_crowdpose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .target_generators.target_generators import GraphGenerator
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator


def build_dataset(cfg, set):
    ok = (set in ['train', 'val'])
    assert ok, 'Please only use build_dataset for training / val.'

    transforms = build_transforms(cfg, ok)

    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
    )
    offset_generator = OffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS
    )

    graph_generator = GraphGenerator(stride=4)       # fixme stride=4?

    dataset = BeanKeypoints(
        cfg,
        set,
        heatmap_generator,
        offset_generator,
        graph_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, 'train')

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_val_dataloader(cfg):

    dataset = build_dataset(cfg, 'val')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return data_loader


def make_test_dataloader(cfg):
    dataset = BeanDataset(
        cfg, cfg.DATASET.TEST
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset



def make_output_dataloader(cfg):
    dataset = OutputDataset(
        cfg, cfg.DATASET.TEST
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset