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

import copy
import glob
from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

import pycocotools
from pycocotools.cocoeval import COCOeval
from lib.utils import zipreader
from lib.utils.rescore import COCORescoreEval

from lib.dataset.beaneval import BEANeval

logger = logging.getLogger(__name__)


class CocoDataset(Dataset):
    def __init__(self, cfg, dataset):
        from pycocotools.coco import COCO
        self.root = cfg.DATASET.ROOT
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        dataset = 'train2017' if 'rescore' in self.dataset else self.dataset
        if 'test' in self.dataset:
            return os.path.join(
                self.root,
                'annotations',
                'image_info_{}.json'.format(
                    self.dataset
                )
            )
        else:
            return os.path.join(
                self.root,
                'annotations',
                'person_keypoints_{}.json'.format(
                    dataset
                )
            )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        dataset = 'train2017' if 'rescore' in self.dataset else self.dataset
        if self.data_format == 'zip':
            return os.path.join(images_dir, dataset) + '.zip@' + file_name
        else:
            return os.path.join(images_dir, dataset, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        image_info = coco.loadImgs(img_id)[0]

        file_name = image_info['file_name']

        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if 'train' in self.dataset:
            return img, [obj for obj in target], image_info
        else:
            return img

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args:
        :param kwargs:
        :return:
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset+tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name[-16:-4])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name[-16:-4]),
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        if 'test' not in self.dataset:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder
            )
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 17

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox_straight': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75',
                       'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str


class BeanDataset_original(Dataset):
    def __init__(self, cfg, dataset):
        self.root = cfg.DATASET.ROOT #os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.reg_thre = cfg.TEST.REG_SCORE_THRE

        self.data_root = self.root #os.path.join(self.root, 'data/bean_combined')

        with open(self._get_anno_file_name()) as f:
            self.anns = json.load(f)
        self.anns = self.anns['annotations']


    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        if 'train' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2023v1lib2_train.json'
            )
        elif 'val' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2023v1lib2_val.json'
            )
        elif 'test' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2023v1lib2_val.json'
            )
        elif 'oks' in self.dataset:
            return os.path.join(
                self.root,
                'bean/annotations',
                'all_annotations_oks.json'
            )


    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.data_root, 'images')
        if 'train' in self.dataset:
            dataset = 'train'
        elif 'val' in self.dataset:
            dataset = 'val'
        elif 'test' in self.dataset:
            dataset = 'test'
        elif 'oks' in self.dataset:
            dataset = 'oks_test'
        else:
            raise FileNotFoundError('Please specify correct dataset name in config file')
        return os.path.join(images_dir, dataset, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        ann = self.anns[index]
        img_name = ann['image_name']
        targets = ann['annotations']

        img = cv2.imread(
            self._get_image_path(img_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        height, width, channels = img.shape

        image_info = {'height': height,
                    'width': width}

        if 'train' in self.dataset or 'val' in self.dataset:
            return img, [obj for obj in targets], image_info
        else:
            return img

    def __len__(self):
        return len(self.anns)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args:
        :param kwargs:
        :return:
        '''
        res_folder = os.path.join(output_dir, 'results')
        self.output_dir = output_dir
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset + tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        oks_nmsed_kpts = []

        for idx, _kpts in enumerate(preds):
            # for an image
            img_name = self.anns[idx]['image_name']
            det_in_image = []
            for idx_kpt, kpt in enumerate(_kpts):
                kpt = self.processKeypoints(kpt)

                detection = {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': img_name,
                    }
                det_in_image.append(detection)
            oks_nmsed_kpts.append(det_in_image)

        self._write_bean_keypoint_results(
            oks_nmsed_kpts, res_file
        )
        # if 'test' not in self.dataset:
        self._do_python_keypoint_eval(res_file)


    def _write_bean_keypoint_results(self, keypoints, res_file):
        # filter by minimum regression score threshold (btw why are there neg values??)

        for kps in keypoints:
            pod_idx_to_remove = []
            if len(kps) == 0:
                continue
            img_name = kps[0]['image']
            for i, pod in enumerate(kps):
                rem_idx = np.where(pod['keypoints'][:, 2] < self.reg_thre)[0]
                if len(rem_idx) == 5:
                    pod_idx_to_remove.append(i)
                    continue
                for idx in rem_idx:
                    pod['keypoints'][idx] = [0, 0, 0]
            for idx in reversed(pod_idx_to_remove):
                kps.pop(idx)

            # visualize
            img_path = self._get_image_path(img_name)
            oriImg = cv2.imread(img_path)
            out = draw_pods(oriImg, kps)
            vis_dir = os.path.join(self.output_dir, 'visualization')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vis_path = os.path.join(vis_dir, img_name)
            cv2.imwrite(vis_path, out)

        results = self._bean_keypoint_results_one_category_kernel(keypoints)
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _bean_keypoint_results_one_category_kernel(self, keypoints):
        cat_results = []
        num_joints = 5

        for img_kpts in keypoints:
            # detections in one image

            # no detection in the image, continue
            if len(img_kpts) == 0:
                continue

            # extract all detections in the image
            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            # (num_detection, 15), rearrangement
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            # column: |1st_x|1st_y|1st_score|2nd_x|2nd_y|2nd_score|...|5th_x|5th_y|5th_score|
            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                # append one detection
                cat_results.append({
                    'image_name': img_kpts[k]['image'],
                    'category_id': 1,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                })

        return cat_results

    def _do_python_keypoint_eval(self, detection_file):
        if 'val' in self.dataset:
            dir_path = os.path.join(self.data_root, 'val')
        elif 'test' in self.dataset:
            dir_path = os.path.join(self.data_root, 'images/test')
        elif 'oks' in self.dataset:
            dir_path = os.path.join(self.data_root, 'images/oks_test')

        img_paths, ann_paths = get_soybean_dataset(dir_path)

        beanGt = ann_paths  # load annotations
        beanDt = loadRes(detection_file)  # load model outputs

        # running evaluation
        beanEval = BEANeval(beanGt, beanDt, mode='nbBean')  # mode: nbBean or area
        beanEval.evaluate()
        beanEval.accumulate()
        beanEval.summarize()

        precisions = beanEval.eval['precision']

        pr_array1 = precisions[0, :, 0, 0, 0]
        print('OKS=0.50:\n', pr_array1)


class BeanDataset(Dataset):
    def __init__(self, cfg, dataset):
        from pycocotools.coco import COCO
        self.root = cfg.DATASET.ROOT #os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.reg_thre = cfg.TEST.REG_SCORE_THRE

        self.data_root = self.root #os.path.join(self.root, 'data/bean_combined')

        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        if 'train' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2022v3_train.json'
            )
        elif 'val' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2022v3_val.json'
            )
        elif 'test' in self.dataset:
            return os.path.join(
                self.data_root,
                'annos',
                'bean_detections_2022v3_val.json'
            )
        elif 'oks' in self.dataset:
            return os.path.join(
                self.root,
                'bean/annotations',
                'all_annotations_oks.json'
            )


    def _get_image_path(self, file_name):
        dataset = 'val' if 'val' in self.dataset else self.dataset
        return os.path.join(self.root, dataset, 'images', file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        targets = coco.loadAnns(ann_ids)

        img_name = coco.loadImgs(img_id)[0]['file_name']
        # ann = self.anns[index]
        # img_name = ann['image_name']
        # targets = ann['annotations']

        img = cv2.imread(
            self._get_image_path(img_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        height, width, channels = img.shape

        image_info = {'height': height,
                    'width': width}

        if 'train' in self.dataset or 'val' in self.dataset:
            return img, [obj for obj in targets], image_info
        else:
            return img

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args:
        :param kwargs:
        :return:
        '''
        res_folder = os.path.join(output_dir, 'results')
        self.output_dir = output_dir
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset + tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        oks_nmsed_kpts = []

        for idx, _kpts in enumerate(preds):
            # for an image
            # img_name = self.anns[idx]['image_name']
            img_name = self.coco.loadImgs(self.ids[idx])[0]['file_name']
            det_in_image = []
            for idx_kpt, kpt in enumerate(_kpts):
                kpt = self.processKeypoints(kpt)

                detection = {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': img_name,
                    }
                det_in_image.append(detection)
            oks_nmsed_kpts.append(det_in_image)

        self._write_bean_keypoint_results(
            oks_nmsed_kpts, res_file
        )
        # if 'test' not in self.dataset:
        self._do_python_keypoint_eval(res_file)


    def _write_bean_keypoint_results(self, keypoints, res_file):
        # filter by minimum regression score threshold (btw why are there neg values??)

        for kps in keypoints:
            pod_idx_to_remove = []
            if len(kps) == 0:
                continue
            img_name = kps[0]['image']
            for i, pod in enumerate(kps):
                rem_idx = np.where(pod['keypoints'][:, 2] < self.reg_thre)[0]
                if len(rem_idx) == 5:
                    pod_idx_to_remove.append(i)
                    continue
                for idx in rem_idx:
                    pod['keypoints'][idx] = [0, 0, 0]
            for idx in reversed(pod_idx_to_remove):
                kps.pop(idx)

            # visualize
            img_path = self._get_image_path(img_name)
            oriImg = cv2.imread(img_path)
            out = draw_pods(oriImg, kps)
            vis_dir = os.path.join(self.output_dir, 'visualization')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vis_path = os.path.join(vis_dir, img_name)
            cv2.imwrite(vis_path, out)

        results = self._bean_keypoint_results_one_category_kernel(keypoints)
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _bean_keypoint_results_one_category_kernel(self, keypoints):
        cat_results = []
        num_joints = 5

        for img_kpts in keypoints:
            # detections in one image

            # no detection in the image, continue
            if len(img_kpts) == 0:
                continue

            # extract all detections in the image
            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            # (num_detection, 15), rearrangement
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            # column: |1st_x|1st_y|1st_score|2nd_x|2nd_y|2nd_score|...|5th_x|5th_y|5th_score|
            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                # append one detection
                cat_results.append({
                    'image_name': img_kpts[k]['image'],
                    'category_id': 1,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                })

        return cat_results

    def _do_python_keypoint_eval(self, detection_file):
        if 'val' in self.dataset:
            dir_path = os.path.join(self.data_root, 'val/images')
        elif 'test' in self.dataset:
            dir_path = os.path.join(self.data_root, 'test/images')
        elif 'oks' in self.dataset:
            dir_path = os.path.join(self.data_root, 'images/oks_test')

        img_paths, ann_paths = get_soybean_dataset(dir_path)

        beanGt = ann_paths  # load annotations
        beanDt = loadRes(detection_file)  # load model outputs

        # running evaluation
        beanEval = BEANeval(beanGt, beanDt, mode='nbBean')  # mode: nbBean or area
        beanEval.evaluate()
        beanEval.accumulate()
        beanEval.summarize()

        precisions = beanEval.eval['precision']

        pr_array1 = precisions[0, :, 0, 0, 0]
        print('OKS=0.50:\n', pr_array1)


class OutputDataset(Dataset):
    def __init__(self, cfg, dataset):
        self.root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data')
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.reg_thre = cfg.TEST.REG_SCORE_THRE

        self.img_dir = os.path.join(self.root, 'bean/images/seperated2')

        self.all_img_name = os.listdir(self.img_dir)

    def _get_image_path(self, file_name):
        return os.path.join(self.img_dir, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        img_name = self.all_img_name[index]
        img = cv2.imread(
            self._get_image_path(img_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape
        return img

    def __len__(self):
        return len(self.all_img_name)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def inference(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args:
        :param kwargs:
        :return:
        '''
        res_folder = os.path.join(output_dir, 'results')
        self.output_dir = output_dir
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset + tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        oks_nmsed_kpts = []

        for idx, _kpts in enumerate(preds):
            # for an image
            img_name = self.all_img_name[idx]
            det_in_image = []
            for idx_kpt, kpt in enumerate(_kpts):
                kpt = self.processKeypoints(kpt)

                detection = {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': img_name,
                    }
                det_in_image.append(detection)
            oks_nmsed_kpts.append(det_in_image)

        self._write_bean_keypoint_results(
            oks_nmsed_kpts, res_file
        )


    def _write_bean_keypoint_results(self, keypoints, res_file):
        # filter by minimum regression score threshold (btw why are there neg values??)

        for kps in keypoints:
            pod_idx_to_remove = []
            if len(kps) == 0:
                continue
            img_name = kps[0]['image']
            for i, pod in enumerate(kps):
                rem_idx = np.where(pod['keypoints'][:, 2] < self.reg_thre)[0]
                if len(rem_idx) == 5:
                    pod_idx_to_remove.append(i)
                    continue
                for idx in rem_idx:
                    pod['keypoints'][idx] = [0, 0, 0]
            for idx in reversed(pod_idx_to_remove):
                kps.pop(idx)

            # visualize
            img_path = self._get_image_path(img_name)

            oriImg = cv2.imread(img_path)
            out = draw_pods(oriImg, kps)
            vis_dir = os.path.join(self.output_dir, 'visualization')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vis_path = os.path.join(vis_dir, img_name)
            cv2.imwrite(vis_path, out)

        results = self._bean_keypoint_results_one_category_kernel(keypoints)
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _bean_keypoint_results_one_category_kernel(self, keypoints):
        cat_results = []
        num_joints = 5

        for img_kpts in keypoints:
            # detections in one image

            # no detection in the image, continue
            if len(img_kpts) == 0:
                continue

            # extract all detections in the image
            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            # (num_detection, 15), rearrangement
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            # column: |1st_x|1st_y|1st_score|2nd_x|2nd_y|2nd_score|...|5th_x|5th_y|5th_score|
            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                # append one detection
                cat_results.append({
                    'image_name': img_kpts[k]['image'],
                    'category_id': 1,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                })

        return cat_results




class CocoRescoreDataset(CocoDataset):
    def __init__(self, cfg, dataset):
        CocoDataset.__init__(self, cfg, dataset)
        
    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset+tag))

        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name[-16:-4])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name[-16:-4]),
                        'area': area
                    }
                )

        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        self._do_python_keypoint_eval(
            cfg.RESCORE.DATA_FILE, res_file, res_folder
        )

    def _do_python_keypoint_eval(self, data_file, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCORescoreEval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.dumpdataset(data_file)




def get_soybean_dataset_original(data_dir):
    # get all img paths and anns paths
    imgs, anns = [], []
    for root, dir, files in os.walk(data_dir):
        files = sorted(files)
        i = 0
        while i < len(files):
            if i + 1 == len(files):
                i += 1
                continue
            name_img, ext_img = os.path.splitext(files[i])
            name_ann, ext_ann = os.path.splitext(files[i + 1])
            # print(name_img, name_ann)

            if ext_img == '.jpg' and ext_ann == '.json' and name_img == name_ann:
                imgs.append(os.path.join(root, files[i]))
                anns.append(os.path.join(root, files[i + 1]))
            i += 2

    return imgs, anns

def get_soybean_dataset(data_dir):
    # get all img paths and anns paths
    imgs, anns = [], []
    for img_file in glob.glob(os.path.join(data_dir, '*.jpg')):
        anno_file = img_file.replace('/images/', '/jsons/')[:-4] + '.json'
        if not os.path.exists(anno_file):
            print('<nofile>{}'.format(anno_file))
            continue
        imgs.append(img_file)
        anns.append(anno_file)

    return imgs, anns





def add_bbox_info(anns, margin=15, shape=None):
    """
    compute bbox information and add it to the annotation
    :param anns: list of dictionaries, each stores annotation for one pod
    :param margin: bounding box margin added to original box
    :param shape: the shape (h, w) of the original image
    :returns : the updated annotation
    """
    anns = copy.deepcopy(anns)
    for id, ann in enumerate(anns):
        s = ann['keypoints']
        x = s[0::3]
        y = s[1::3]
        z = s[2::3]
        idx = [i for i, v in enumerate(z) if v > 0]

        nb_points = np.count_nonzero(z)
        first_bean_pos = np.array([x[idx[0]], y[idx[0]]])
        last_bean_pos = np.array([x[idx[-1]], y[idx[-1]]])
        direction = last_bean_pos - first_bean_pos
        dist = np.sqrt(np.sum(direction ** 2)) + 1e-8
        direction = direction / dist
        direction_vertical = np.array([direction[1], -direction[0]])

        if nb_points > 1:
            pos1 = first_bean_pos - direction * (margin*2) + direction_vertical * margin
            pos2 = first_bean_pos - direction * (margin*2) - direction_vertical * margin
            pos3 = last_bean_pos + direction * (margin*2) - direction_vertical * margin
            pos4 = last_bean_pos + direction * (margin*2) + direction_vertical * margin
            polygon = list(map(list, [pos1, pos2, pos3, pos4]))
            polygon = np.array(polygon)

        elif nb_points == 1:
            x0 = max(0, first_bean_pos[0] - margin)
            y0 = max(0, first_bean_pos[1] - margin)
            x1 = round(first_bean_pos[0]) + margin
            y1 = round(first_bean_pos[1]) + margin

            polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        # clip values to valid range
        if shape:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1])
            polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1])
            polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0])
            polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0])

        # define box area for bean pods
        h = np.sqrt(np.sum((polygon[1] - polygon[2]) ** 2))
        ann['area'] = margin * 2 * h
        ann['bbox'] = np.round(np.array(polygon)).astype(int)
    return anns

def loadRes(res_file):
    """
    Load temporary result file
    :param res_file: the path of temporary result json file
    :returns : the regularized annotation
    """
    with open(res_file) as f:
        anns = json.load(f)
    assert type(anns) == list, 'results is not an array of objects'
    return add_bbox_info(anns)


def draw_pods(npimg, anns, imgcopy=False):
    """
    draw keypoints and connection lines on the image
    """

    BeanColors = [
        [51, 51, 255], [51, 153, 255], [51, 255, 255], [51, 255, 153], [255, 153, 51]]
    BeanPairsRender = [
        (0, 1), (1, 2), (2, 3), (3, 4)
    ]
    BeanLineColors = [[255, 153, 153], [255, 153, 204], [255, 255, 204], [204, 255, 229]]

    if imgcopy:
        npimg = np.copy(npimg)

    for id, ann in enumerate(anns):
        centers = {}
        s = ann['keypoints']
        nb_keypoints = len(s)
        x = s[:, 0]
        y = s[:, 1]
        z = s[:, 2]

        for i in range(nb_keypoints):
            if z[i] != 0:
                center = np.round(x[i]).astype(int), np.round(y[i]).astype(int)
                cv2.circle(npimg, center, 3, BeanColors[i], thickness=3, lineType=8, shift=0)
                centers[i] = center

        # draw line
        for pair_order, pair in enumerate(BeanPairsRender):
            if not z[pair[0]] or not z[pair[1]]:
                continue

            cv2.line(npimg, centers[pair[0]], centers[pair[1]], BeanLineColors[pair_order], 3)

    return npimg