# coding: utf-8
# @email: enoche.chow@gmail.com

"""
MMTSF quick start 
##########################
"""
from cmath import inf
from logging import getLogger
from itertools import product
import torch.multiprocessing as mp
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, dict2str
import platform
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any
import multiprocessing
import torch
from tqdm import tqdm
from common.trainer import Trainer 
from common.dataloader import data_loader
import warnings
import json
from collections import defaultdict

warnings.filterwarnings("ignore")  # 忽略所有警告


def convert_float32_to_float(obj):
    """递归将numpy.float32转换为python float"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_float32_to_float(x) for x in obj]
    return obj

def run_model(args):
    """
    封装模型训练和获取指标的过程
    :param args: 包含配置信息、总循环次数、当前循环索引、数据加载器、模型、训练器等的元组
    :return: 最佳验证分数、最佳验证结果、最佳测试结果
    """
    config, hyper_tuple, total_loops, idx, train_loader, valid_loader, test_loader, model, save_model, mg, logger = args
    init_seed(config['seed'])
    logger.info('========={}/{}: Parameters:{}={}======='.format(
        idx+1, total_loops, config['hyper_parameters'], hyper_tuple))
    
    # model loading and initialization
    model = get_model(config['model'])(config).to(config['device'])
    model = model.float() 
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(model, config)
    
    # model training
    best_valid_score, best_test_upon_valid = trainer.fit(
        train_loader, valid_loader=valid_loader, test_loader=test_loader, saved=save_model)
    
    # 转换结果中的float32为float
    best_test_upon_valid = convert_float32_to_float(best_test_upon_valid)
    
    return config['hyper_parameters'], best_test_upon_valid, hyper_tuple

def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(config_dict, model, dataset)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    train_loader, valid_loader, test_loader = data_loader(config=config)
    
    ############ run model
    hyper_ret = []
    best_test_value = inf
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # 加载已有的结果文件
    existing_results = {}
    # json_file = f'''{model}_{dataset}.json'''
    # json_file = '3output.json' #先注释掉了，防止不小心别的进程写进去
    logger.info(f"The loading and write to json file is: {json_file}=================================\n")
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, 'r') as f:
                existing_results = json.load(f)
            logger.info(f"Loaded {len(existing_results)} existing results from {json_file}")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error loading existing results: {str(e)}")
            existing_results = {}

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])

    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    config_list = []
    hyper_tuple_list = []
    skipped_count = 0
    
    import copy
    for hyper_tuple in combinators:
        # 创建配置字典
        new_config = copy.deepcopy(config)
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            new_config[j] = k
        
        # 检查是否已存在相同配置的结果
        param_dict = dict(zip(config['hyper_parameters'], hyper_tuple))
        safe_param_dict = convert_float32_to_float(param_dict)
        param_str = str(safe_param_dict)
        
        if param_str in existing_results:
            skipped_count += 1
            logger.info(f'Skipping existing parameters: {param_str}')
            continue
            
        config_list.append(new_config)
        hyper_tuple_list.append(hyper_tuple)

    logger.info(f"Total combinations: {total_loops}, Skipped: {skipped_count}, Remaining: {len(config_list)}")

    args_list = []
    for idx, (config_x, hyper_tuple_x) in enumerate(zip(config_list, hyper_tuple_list)):
        args = (config_x, hyper_tuple_x, total_loops, idx, train_loader, valid_loader, test_loader, model, save_model, mg, logger)
        args_list.append(args)
    
    with tqdm(total=len(args_list), desc="Processing Hyperparameter Combinations") as pbar:
        with mp.Pool(processes=config['processes_num']) as pool:
            for result_idx, result in enumerate(pool.imap_unordered(run_model, args_list)):
                p, best_test_upon_valid, hyper_tuple = result
                hyper_ret.append((hyper_tuple, best_test_upon_valid))
                logger.info('========={}/{}: Parameters:{}={}======='.format(
                    result_idx+1, total_loops, config['hyper_parameters'], hyper_tuple))
                
                # 确保所有数据都是JSON可序列化的
                safe_test_result = convert_float32_to_float(best_test_upon_valid)
                param_dict = dict(zip(p, [config_list[hyper_ret.index((hyper_tuple, best_test_upon_valid))][param] for param in p]))
                safe_param_dict = convert_float32_to_float(param_dict)
                
                result_dict = {
                    str(safe_param_dict): {
                        "best_test_result": safe_test_result
                    }
                }
                
                # 更新现有结果并保存
                existing_results.update(result_dict)
                
                # 原子写入文件
                temp_file = json_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(existing_results, f, indent=4)
                os.replace(temp_file, json_file)
                
                # 更新最佳结果索引
                if safe_test_result["MSE"] < best_test_value:
                    best_test_value = safe_test_result["MSE"]
                    best_test_idx = result_idx
                    
                logger.info('test result: {}'.format(dict2str(safe_test_result)))
                
                if hyper_ret:
                    logger.info('████Current BEST████:\nParameters: {}={},\n'
                            'Test: {}\n\n\n'.format(config['hyper_parameters'],
                            hyper_ret[best_test_idx][0], 
                            dict2str(hyper_ret[best_test_idx][1])))
                else:
                    logger.warning("No valid results yet")
                    
                pbar.update(1)

    # log info
    logger.info('\n============All Over=====================')
    for (p, k) in hyper_ret:
        logger.info('Parameters: {}={},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1])))