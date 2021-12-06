import argparse
import importlib
import os
import os.path as osp
import time
import json

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.utils import (get_root_logger, print_log, traverse_replace)


def nondist_single_forward_collect(func, data_loader, length):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(list)): The concatenated outputs.
    """
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if k == 'intra_bbox':
            intra_results_list = [batch[k].numpy().tolist() for batch in results]
            results_all[k] = intra_results_list
            assert len(results_all[k]) == length
    inter_results_list = []
    for batch in results:
        merge_batch_results = []
        for k in batch.keys():
            if k != 'intra_bbox':
                merge_batch_results.append(batch[k].numpy().tolist())
        inter_results_list.append(merge_batch_results)
    results_all['inter_bbox'] = inter_results_list
    assert len(results_all['inter_bbox']) == length
    return results_all


def dist_single_forward_collect(func, data_loader, rank, length):
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        rank (int): This process id.

    Returns:
        results_all (dict(list)): The concatenated outputs.
    """
    results = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)  # dict{key: tensor}
        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if k == 'intra_bbox':
            intra_results_list = [batch[k].numpy().tolist() for batch in results]
            results_all[k] = intra_results_list
    inter_results_list = []
    for batch in results:
        merge_batch_results = []
        for k in batch.keys():
            if k != 'intra_bbox':
                merge_batch_results.append(batch[k].numpy().tolist())
        inter_results_list.append(merge_batch_results)
    results_all['inter_bbox'] = inter_results_list
    return results_all


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_single_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_single_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate correspondence in Stage 2 of ORL')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'input', type=str, help='input knn instance json file')
    parser.add_argument(
        'output',
        type=str,
        help='output correspondence json file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
        cfg.work_dir = args.work_dir

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'test_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: list}

    rank, _ = get_dist_info()
    if rank == 0:
        out = dataset.evaluate(
            args.input, **outputs)
        with open(args.output, 'w') as f:
            json.dump(out, f)
        print_log("Correspondence json file has been saved to: {}".format(
            args.output), logger=logger)


if __name__ == '__main__':
    main()
