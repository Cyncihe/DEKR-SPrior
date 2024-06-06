# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import numpy as np
sys.path.append(os.path.abspath("./"))

import _init_paths
import lib.models as models
from lib.models import hrnet_dekr_graph, hrnet_dekr

import matplotlib.pyplot as plt
import matplotlib.cm as CM

from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_multi_stage_outputs
from lib.core.inference import aggregate_results
from lib.core.nms import pose_nms
from lib.core.match import match_pose_to_heatmap
from lib.utils.utils import create_logger
from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size
from lib.utils.rescore import rescore_valid
from tools.papers.show_heamap_on_image import apply_colormap_on_image,apply_colormap_on_image_all
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append(os.path.abspath("./"))

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class BeanModel(object):
    def __init__(self, model_path):
        parser = argparse.ArgumentParser(description='Test keypoints network')
        # general
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            default=os.path.join(ROOT,
                                                 'experiments/bean/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140.yaml'),
                            type=str)

        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)

        args = parser.parse_args()

        self.logger, self.final_output_dir, _ = create_logger(
            cfg, args.cfg, 'valid'
        )

        self.logger.info(pprint.pformat(args))
        self.logger.info(cfg)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        update_config(cfg, args)
        self.model = self.create_model(model_path)

        # data_loader, test_dataset = make_test_dataloader(cfg)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def create_model(self, model_path):
        model = eval(cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )
        self.logger.info('=> loading model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        model.eval()
        return model

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def inference(self, image_rgb, img_name=None):
        # assert 1 == images.size(0), 'Test batch size should be 1'
        # image = images[0].cpu().numpy()
        _input_h, _input_w, _input_channel = image_rgb.shape
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image_rgb, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            heatmap_sum = 0
            poses = []

            for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                image_resized, center, scale_resized = resize_align_multi_scale(
                    image_rgb, cfg.DATASET.INPUT_SIZE, scale, 1.0
                )

                image_resize_original = image_resized.copy()

                image_resized = self.transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                heatmap, posemap, offset = get_multi_stage_outputs(
                    cfg, self.model, image_resized, cfg.TEST.FLIP_TEST
                )
                heatmap_sum, poses = aggregate_results(
                    cfg, heatmap_sum, poses, heatmap, posemap, scale
                )
            
            heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)

            if img_name is not None:
                # heat1, heat2 = apply_colormap_on_image(image_resize_original, heatmap_avg[0][0].cpu().numpy(),
                #                                        'Greens', alpha=0.6)
                heat2 = apply_colormap_on_image_all(image_resize_original, heatmap_avg[0].cpu().numpy(),
                                                    alpha=0.6, type=3)
                save_path = os.path.join('/data1/datas/beans/papers/test_crop/results', img_name[:-4] + '.png')
                print('save file:{}'.format(save_path))
                heat2.save(save_path)

                plt.figure(figsize=(16, 16), dpi=100)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                for i in range(10):
                    plt.imshow(offset[0][i].detach().cpu().numpy(), cmap=CM.jet)
                    plt.xticks([])
                    plt.yticks([])
                    save_path = os.path.join('/data1/datas/beans/papers/test_crop/results', img_name[:-4] + '_pose{}'.format(i) + '.jpg')
                    plt.savefig(save_path, pad_inches=0)
                plt.imshow(heat2)
                plt.show()
                # cv2.imwrite('./results/img1.jpg', heat2[:,:,::-1])
            poses, scores = pose_nms(cfg, heatmap_avg, poses)
            final_poses = []
            if len(scores) > 0:
                if cfg.TEST.MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

                final_poses = get_final_preds(
                    poses, center, scale_resized, base_size
                )
                if cfg.RESCORE.VALID:
                    scores = rescore_valid(cfg, final_poses, scores)

            # add on 2022.11.07 by hjj
            # out to original image size
            # aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

            # height, width = image_resize_original.shape[:2]
            #
            # center = np.array((width / 2, height / 2))
            # scale = min(height, width) / 200
            #
            #
            # mat_output = self._get_affine_matrix(
            #     center, scale, (_input_w, _input_h), 0
            # )[:2]
            #
            # image_out = cv2.warpAffine(
            #     image_resize_original,
            #     mat_output,
            #     (_input_w, _input_h)
            # )
            # for i,p in enumerate(poses):
            #     poses[i][:, :, 0:2] = self._affine_joints(
            #         poses[i][:, :, 0:2], mat_output
            #     )
            return final_poses, scores


