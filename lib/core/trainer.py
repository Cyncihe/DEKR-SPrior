# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

import torch

from lib.utils.comp_graph import graph_metirc_gt, graph_metirc_dt, graph_batch_l2_loss
from lib.utils.utils import AverageMeter


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, (image, heatmap, mask, offset, offset_w, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        pheatmap, poffset = model(image)

        heatmap = heatmap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True)
        offset_w = offset_w.cuda(non_blocking=True)

        heatmap_loss, offset_loss = \
            loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)

        loss = 0
        if heatmap_loss is not None:
            heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
            loss = loss + heatmap_loss
        if offset_loss is not None:
            offset_loss_meter.update(offset_loss.item(), image.size(0))
            loss = loss + offset_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(
                          heatmap_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offset')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1


def do_validate(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict):

    logger = logging.getLogger("Validation")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():

        end = time.time()
        for i, (image, heatmap, mask, offset, offset_w, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            pheatmap, poffset = model(image)

            heatmap = heatmap.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            offset_w = offset_w.cuda(non_blocking=True)

            heatmap_loss, offset_loss = \
                loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)


            loss = 0
            if heatmap_loss is not None:
                heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
                loss = loss + heatmap_loss
            if offset_loss is not None:
                offset_loss_meter.update(offset_loss.item(), image.size(0))
                loss = loss + offset_loss

            total_loss_meter.update(loss.item(), image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                epoch, i, len(data_loader),
                batch_time=batch_time,
                speed=image.size(0) / batch_time.val,
                data_time=data_time,
                heatmaps_loss=_get_loss_info(
                    heatmap_loss_meter, 'heatmaps'),
                offset_loss=_get_loss_info(offset_loss_meter, 'offset')
            )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1

    return total_loss_meter.avg

def do_train_graph(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, graph_lambda):
    print('graph_lambda:', graph_lambda)
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, (image, heatmap, mask, offset, offset_w, group_mask) in enumerate(data_loader):
        data_time.update(time.time() - end)

        pheatmap, poffset, graph_features = model(image)

        heatmap = heatmap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True)
        offset_w = offset_w.cuda(non_blocking=True)
        group_mask = group_mask.cuda(non_blocking=True)

        heatmap_loss, offset_loss = \
            loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)

        graph_loss = get_graph_loss(group_mask, graph_features)

        loss = 0
        if heatmap_loss is not None:
            heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
            loss = loss + heatmap_loss
        if offset_loss is not None:
            offset_loss_meter.update(offset_loss.item(), image.size(0))
            loss = loss + offset_loss

        # print("DEKR_loss:", loss)
        # print("graph_loss:", graph_lambda * graph_loss)

        loss = loss + graph_lambda * graph_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(
                          heatmap_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offset')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1



def do_validate_graph(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, graph_lambda):

    logger = logging.getLogger("Validation")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()
    graph_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (image, heatmap, mask, offset, offset_w, group_mask) in enumerate(data_loader):
            data_time.update(time.time() - end)

            pheatmap, poffset, graph_features = model(image)

            heatmap = heatmap.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            offset_w = offset_w.cuda(non_blocking=True)
            group_mask = group_mask.cuda(non_blocking=True)

            heatmap_loss, offset_loss = \
                loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)

            graph_loss = get_graph_loss(group_mask, graph_features)


            graph_loss_meter.update(graph_loss.item(), image.size(0))

            loss = 0
            if heatmap_loss is not None:
                heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
                loss = loss + heatmap_loss
            if offset_loss is not None:
                offset_loss_meter.update(offset_loss.item(), image.size(0))
                loss = loss + offset_loss

            # print("DEKR_loss:", loss)
            # print("graph_loss:", graph_lambda * graph_loss)

            loss = loss + graph_lambda * graph_loss
            total_loss_meter.update(loss.item(), image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{heatmaps_loss}{offset_loss}{graph_loss}'.format(
                    epoch, i, len(data_loader),
                    batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time,
                    heatmaps_loss=_get_loss_info(
                        heatmap_loss_meter, 'heatmaps'),
                    offset_loss=_get_loss_info(offset_loss_meter, 'offset'),
                    graph_loss=_get_loss_info(graph_loss_meter, 'graph')
                )
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar(
                    'train_heatmap_loss',
                    heatmap_loss_meter.val,
                    global_steps
                )
                writer.add_scalar(
                    'train_offset_loss',
                    offset_loss_meter.val,
                    global_steps
                )
                writer_dict['train_global_steps'] = global_steps + 1

    return total_loss_meter.avg



def _get_loss_info(meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=meter
    )

    return msg




def get_graph_loss(group_mask, graph_features):
    gt_mtx = graph_metirc_gt(group_mask)  # gt_mtx[0].cpu().numpy()
    dt_mtx = graph_metirc_dt(graph_features, group_mask)
    graph_loss = graph_batch_l2_loss(dt_mtx, gt_mtx)
    return graph_loss
