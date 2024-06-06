# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings
import sys

import numpy as np
sys.path.append(os.path.abspath("../"))

from lib.dataset.build import make_val_dataloader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
from lib.models import hrnet_dekr_graph, hrnet_dekr

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import MultiLossFactory
from lib.core.trainer import do_train, do_train_graph, do_validate_graph, do_validate
from lib.dataset import make_dataloader
from lib.utils.utils import create_logger
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import setup_logger
from lib.utils.utils import get_model_summary


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=os.path.join(ROOT, 'experiments/bean/w32/graph_w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140.yaml'),
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        # default='',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:1144',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"
    print("Use GPU: {} for training".format(os.environ['CUDA_VISIBLE_DEVICES']))

    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    print('graph_lambda:', cfg.TRAIN.GRAPH_LAMBDA)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    # logger.info(cfg)
    print(cfg.GPUS)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:

        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir
        )


def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    model = eval(cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )
    val_loader = make_val_dataloader(cfg)

    logger.info(train_loader.dataset)

    best_val_loss = np.inf
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, 
                        map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train one epoch
        if 'graph' in cfg.MODEL.NAME:
            do_train_graph(cfg, model, train_loader, loss_factory, optimizer, epoch,
                     final_output_dir, tb_log_dir, writer_dict, graph_lambda=cfg.TRAIN.GRAPH_LAMBDA)

            val_loss = do_validate_graph(cfg, model, val_loader, loss_factory, optimizer, epoch,
                              final_output_dir, tb_log_dir, writer_dict, graph_lambda=cfg.TRAIN.GRAPH_LAMBDA)
        else:
            do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                     final_output_dir, tb_log_dir, writer_dict)

            val_loss = do_validate(cfg, model, val_loader, loss_factory, optimizer, epoch,
                                         final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()

        perf_indicator = val_loss
        if perf_indicator < best_val_loss:
            best_val_loss = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth.tar'
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
