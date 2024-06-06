# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 15:05
# @Author  : hjj
# @contact : hejingjing@zhejianglab.com
# @Site    : 
# @File    : comp_graph.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np

def graph_l2_loss(input, target):
    loss = (input - target)
    loss = (loss * loss) / 2
    return loss.sum()

def graph_batch_l2_loss(input, target):
    losses = 0
    for n in range(len(input)):
        loss = (input[n] - target[n])
        losses += ((loss * loss) / 2).sum()
    return losses / len(input)
#
# def graph_batch_l2_loss(input, target):
#     losses = 0
#     for n in range(len(input)):
#         row,col = input[n].shape
#         for i in range(row):
#             for j in range(col):
#                 loss = (input[n][i][j] - target[n][i][j])
#                 losses += (loss * loss) / 2
#     return losses


def BCEWithLogitLoss():
    return nn.BCEWithLogitsLoss()

def comp_similarity(loss_func, x, keypoints, group_masks,images,stride=8.):
    N, dim, _, _ = x.shape
    _,_,h,w = images.shape
    loss_graph = [0] * N

    for n in range(N):
        # kpt = keypoints[n]
        # gmask = group_masks[n]

        filter_kpts = []
        filter_group = []
        for i, p in enumerate(keypoints):
            if p[0] < 0 or p[1] < 0 or p[0] >= w or p[1] >= h:
                continue
            filter_kpts.append([p[0].numpy().tolist()[0]/stride,p[1].numpy().tolist()[0]/stride])
            filter_group.append(group_masks[i].numpy().tolist()[0])
        if len(filter_kpts) < 1:
            continue

        filter_kpts = np.array(filter_kpts)
        tmp = -torch.ones((int(h/stride),int(w/stride)))
        index = [torch.LongTensor(filter_kpts[:, 1]), torch.LongTensor(filter_kpts[:, 0])]
        # index = torch.tensor(index)
        group = torch.Tensor(filter_group)
        tmp = tmp.index_put(index, group)
        mask = tmp >= 0

        # gt
        target = torch.masked_select(tmp, mask)
        t0 = target.unsqueeze(1).expand(target.size(0), target.size(0))
        t1 = target.expand(target.size(0), target.size(0))
        gt_metric = (t0 == t1).float().cuda()

        # predict
        x2 = torch.masked_select(x, mask.cuda())
        x2 = x2.reshape(N, dim, -1)
        x2 = x2.transpose(1, 2)
        dt_metric = torch.cosine_similarity(x2.unsqueeze(2), x2.unsqueeze(1), dim=3, eps=1e-8)
        dt_metric = dt_metric.squeeze(0)
        loss_graph[n] = loss_func(dt_metric,gt_metric)

    return loss_graph


def graph_metirc_gt(group_mask):
    N, _, _ = group_mask.shape
    gts_metric = []
    for n in range(N):
        mask = group_mask[n] >= 0
        # gt
        target = torch.masked_select(group_mask[n], mask)
        t0 = target.unsqueeze(1).expand(target.size(0), target.size(0))
        t1 = target.expand(target.size(0), target.size(0))
        gt_metric = (t0 == t1).float()
        gts_metric.append(gt_metric)
    return gts_metric

def graph_metirc_dt(inputX, group_mask):
    # predict
    N, dim, _, _ = inputX.shape
    dts_metric = []
    for n in range(N):
        mask = group_mask[n] >= 0   # (46, 46), position with bean
        # x_unsq = inputX[n].unsqueeze(0)
        x2 = torch.masked_select(inputX[n], mask)  # extract feature vectors
        x2 = x2.reshape(1, dim, -1)
        x2 = x2.transpose(1, 2)
        dt_metric = torch.cosine_similarity(x2.unsqueeze(2), x2.unsqueeze(1), dim=3, eps=1e-8)
        dts_metric.append(dt_metric.squeeze(0))
    return dts_metric


def graph_mask(kpts,size):
    tmp = -np.ones(size)
    for p in kpts:
        tmp[int(p[0]),int(p[1])] = 1
    return tmp > 0


def graph_label(keypoints, group_mask,size,stride):
    # keypoints = np.array(keypoints)
    # size : h,w
    filter_kpts = []
    filter_group = []
    for i,p in enumerate(keypoints):
        if p[0] < 0 or p[1] < 0 or p[0] >= size[1] or p[1] >= size[0]:
            continue
        filter_kpts.append(p)
        filter_group.append(group_mask[i])
    filter_kpts = np.array(filter_kpts)
    # label = gene_label(keypoints, group_mask, size)

    kpts = np.array(filter_kpts) / stride
    tmp = -torch.ones(int(size[0] / stride), int(size[1] / stride))
    index = [torch.LongTensor(kpts[:, 0]), torch.LongTensor(kpts[:, 1])]
    # index = torch.tensor(index)
    group = torch.Tensor(group_mask)
    tmp = tmp.index_put(index, group)
    # mask = tmp >= 0

    return tmp


def generate_graph_groupmap(keypoints, group_mask, img_shape, stride):
    h, w = img_shape
    groupmap = - torch.ones((int(h / stride), int(w / stride)))
    filter_kpts = []
    filter_group = []
    for i, p in enumerate(keypoints):
        if p[0] < 0 or p[1] < 0 or p[0] >= w or p[1] >= h:
            continue
        filter_kpts.append(p)
        filter_group.append(group_mask[i])
    if len(filter_kpts) < 1:
        return groupmap

    filter_kpts = np.array(filter_kpts) / stride
    for i, p in enumerate(filter_kpts):
        p = list(map(int,p))
        groupmap[p[1]][p[0]] = filter_group[i]

    return groupmap


'''
def graph_label(keypoints, group_mask,size,stride):
    label = gene_label(keypoints,group_mask,size)

    kpts = np.array(keypoints) / stride
    tmp = -torch.ones(int(size[0]/stride),int(size[1]/stride))
    index = [torch.LongTensor(kpts[:,0]), torch.LongTensor(kpts[:,1])]
    # index = torch.tensor(index)
    group = torch.Tensor(group_mask)
    tmp = tmp.index_put(index, group)
    mask = tmp >= 0

    return mask, label
'''