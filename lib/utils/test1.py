import argparse
import os

import torch

from lib.dataset.COCOKeypoints import BeanKeypoints
from lib.dataset.target_generators import HeatmapGenerator, OffsetGenerator
from lib.dataset.target_generators.target_generators import GraphGenerator
from lib.dataset.transforms import build_transforms
from lib.models import hrnet_dekr_graph, hrnet_dekr

from lib.config import cfg
from lib.config import update_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=os.path.join(ROOT, 'experiments/bean/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140.yaml'),
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:1434',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args
args = parse_args()
update_config(cfg, args)

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
        cfg.DATASET.TRAIN,
        heatmap_generator,
        offset_generator,
        graph_generator,
        transforms
    )

    return dataset

dataset = build_dataset(cfg, 'val')

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

print(data_loader)