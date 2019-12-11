from __future__ import division

import argparse
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch

# 读取命令行的参数
# 输入 --help，可以看输入的参数格式和内容
# 1个参数必须输入：配置文件(config
# 7个参数可选
# --work_dir:文件输出目录；--resume_from:是否在某个checkpoint的基础上继续训练；--validate:是否对每个checkpoint进行评估，默认为true；--gpus:使用gpu的数量，默认为1；--launcher:分布式训练的任务启动器，默认为none
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    # 读取命令行的参数
    args = parse_args()
    # 读取配置文件
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    # 如果命令行中没有设定工作空间，就按照默认的——work_dir = './work_dirs/cascade_rcnn_r50_fpn_1x'；如果有输入就更新
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 如果是在预训练的基础上继续训练，那就更新cfg，否则就按照默认的resume_from = None
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    # 输入的gpu数量来设置
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    # 如果不设置分布式的，那么distributed的值为false
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # 核心调用 build_detector,get_dataset,train_detector

    # train函数的核心 —— 调用build_detector()来创建模型，将config配置文件中的数据加载到建立的模型中去，返回的是对应的网络实例化的对象
    model = build_detector(
        # 获得config文件的model配置数据，train的配置数据，test的配置数据
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # 注册数据集，获得cfg中的data字典其中的train字段，也为字典类型
    # 返回是一个dict，有数据集相关的数据和datasets所有的数据集标签。
    train_dataset = get_dataset(cfg.data.train)

    # 开始训练
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
