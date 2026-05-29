# coding: utf-8
# @email: enoche.chow@gmail.com

"""
MMTSF quick start 
##########################
"""
from cmath import inf
from logging import getLogger
from itertools import product

from model_trainer.utils.logger import init_logger
from model_trainer.utils.configurator import Config
from model_trainer.utils.utils import init_seed, get_model, dict2str
import platform
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from model_trainer.common.trainer import Trainer
from model_trainer.common.dataloader import data_loader
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告


def quick_start(model, dataset, config_dict, save_model=True):
    config = Config(config_dict, model, dataset)

    distributed = bool(config_dict.get('distributed', False))
    rank = int(config_dict.get('rank', 0))
    world_size = int(config_dict.get('world_size', 1))
    config['distributed'] = distributed
    config['rank'] = rank
    config['world_size'] = world_size

    init_logger(config)
    logger = getLogger()
    is_main = (not distributed) or rank == 0

    if is_main:
        logger.info('██Server: \t' + platform.node())
        logger.info('██Dir: \t' + os.getcwd() + '\n')
        logger.info(config)

    train_loader, valid_loader, test_loader = data_loader(config=config)
    if distributed and rank != 0:
        valid_loader = None
        test_loader = None

    hyper_ret = [] if is_main else None
    best_test_value = inf
    best_test_idx = 0

    if is_main:
        logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for param_name in config['hyper_parameters']:
        value = config[param_name]
        if isinstance(value, list):
            hyper_ls.append(value)
        elif value is None:
            hyper_ls.append([None])
        else:
            hyper_ls.append([value])

    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    
    # ========== 禁用并发：强制串行搜索 ==========
    for idx, hyper_tuple in enumerate(combinators):
        for key, value in zip(config['hyper_parameters'], hyper_tuple):
            config[key] = value
        init_seed(config['seed'])

        if is_main:
            logger.info('========={}/{}: Parameters:{}={}======='.format(
                idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        model_instance = get_model(config['model'])(config).to(config['device']).float()

        if distributed:
            device = config['device']
            device_ids = [device.index] if isinstance(device, torch.device) and device.type == 'cuda' else None
            ddp_kwargs = {
                'device_ids': device_ids,
                'output_device': device.index if device_ids else None,
                'find_unused_parameters': bool(config.get('ddp_find_unused_parameters', False)),
            }
            if distributed and not ddp_kwargs['find_unused_parameters']:
                logger.debug("find_unused_parameters=False, if you see DDP errors about unused parameters, set ddp_find_unused_parameters: true in config")
            model_instance = DDP(model_instance, **ddp_kwargs)

        trainer = Trainer(model_instance, config)

        current_save_flag = save_model and ((not distributed) or rank == 0)
        best_valid_score, best_test_metrics = trainer.fit(
            train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            saved=current_save_flag,
        )

        if not is_main:
            continue

        best_test_metrics = best_test_metrics or {}
        hyper_ret.append((hyper_tuple, best_test_metrics))

        mse_value = best_test_metrics.get("MSE", float('inf'))
        if mse_value < best_test_value:
            best_test_value = mse_value
            best_test_idx = idx

        logger.info('test result: {}'.format(dict2str(best_test_metrics)))
        logger.info('████Current BEST████:\nParameters: {}={},\nTest: {}\n\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            dict2str(best_test_metrics),
        ))

    if not is_main:
        return

    logger.info('\n============All Over=====================')
    for params, metrics in hyper_ret:
        logger.info('Parameters: {}={},\n best test: {}'.format(
            config['hyper_parameters'], params, dict2str(metrics)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1]),
    ))
