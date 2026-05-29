import argparse
import os
from copy import deepcopy
from typing import List

import torch.multiprocessing as mp

from model_trainer.utils.quick_start import quick_start


os.environ.setdefault('NUMEXPR_MAX_THREADS', '48')


def _parse_gpu_list(gpu_string: str) -> List[int]:
    entries = [token.strip() for token in gpu_string.split(',') if token.strip()]
    if not entries:
        raise ValueError('Invalid --gpus argument: no GPU ids found')
    try:
        return [int(entry) for entry in entries]
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f'Invalid GPU id in --gpus: {gpu_string}') from exc


def _distributed_worker(local_rank: int, model_name: str, dataset_name: str, base_config: dict) -> None:
    import torch
    import torch.distributed as dist

    world_size = base_config.get('world_size', torch.cuda.device_count())
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)

    worker_config = deepcopy(base_config)
    worker_config['gpu_id'] = local_rank
    worker_config['rank'] = local_rank
    worker_config['distributed'] = True
    worker_config['world_size'] = world_size
    worker_config.pop('gpu_ids', None)

    try:
        quick_start(
            model=model_name,
            dataset=dataset_name,
            config_dict=worker_config,
            save_model=(local_rank == 0),
        )
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forecasting model with predefined configs.')
    parser.add_argument('--model', '-m', type=str, default='TimeLLM', help='registered model name')
    parser.add_argument('--dataset', '-d', type=str, default='electricity', help='registered dataset name')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device index (single GPU mode)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for Distributed Data Parallel training')
    parser.add_argument('--dataset-alias', type=str, help='dataset alias for specific version')
    parser.add_argument('--gt-embedding-alias', type=str, help='GT embedding dataset alias for loss computation')
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = _parse_gpu_list(args.gpus)
    else:
        gpu_ids = [args.gpu]

    config_dict = {}

    #temp config
    if args.dataset_alias:
        config_dict['dataset_alias'] = args.dataset_alias
    
    if args.gt_embedding_alias:
        config_dict['gt_embedding_alias'] = args.gt_embedding_alias


    mp.set_start_method('spawn', force=True)

    if len(gpu_ids) > 1:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in gpu_ids)

        distributed_config = dict(config_dict)
        distributed_config['gpu_ids'] = gpu_ids
        distributed_config['world_size'] = len(gpu_ids)

        # 设置分布式训练超时时间
        os.environ.setdefault('TORCH_DISTRIBUTED_TIMEOUT', '3600000')
        
        mp.spawn(
            _distributed_worker,
            nprocs=len(gpu_ids),
            args=(args.model, args.dataset, distributed_config),
            join=True,
        )
    else:
        config_dict['gpu_id'] = gpu_ids[0]
        config_dict['distributed'] = False
        quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
