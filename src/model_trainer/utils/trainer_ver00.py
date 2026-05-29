import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from time import time
from logging import getLogger
import torch.optim as optim
from utils.evaluator import TemporalEvaluator
import numpy as np  
from utils.utils import get_local_time, early_stopping, dict2str
import torch.nn as nn

class Trainer:
    def __init__(self, model,config):
        self.logger = getLogger()
        self.config = config
        
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['patience']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric']
        self.model = model
        self.device = config["device"]
        self.inverse = config["inverse"]

        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.req_training = config['req_training']
        self.start_epoch = 0
        self.cur_step = 0
        
        self.best_valid_score = float('inf')
        self.train_loss_dict = dict()

        self.optimizer = self._build_optimizer()
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        lr_scheduler = config['learning_rate_scheduler'] 
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler 
        self.evaluator = TemporalEvaluator(config)
        
        
    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
   
    def train_epoch(self, train_loader,epoch_idx,loss_func=None):
        """单个训练步骤"""
        if not self.req_training:
            return 0.0, []
        
        train_loss = 0
        loss_batches = []
        for batch_idx, (batch_x,batch_prompt,batch_y) in tqdm(enumerate(train_loader), desc="Training"):
            self.optimizer.zero_grad()
     
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x,batch_prompt,flag='train')
            loss = self.model.calculate_loss(batch_y)
            train_loss +=loss
            loss_batches.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            
        
        
        return train_loss,loss_batches
        

    def _valid_epoch(self, valid_loader):
       valid_result = self.evaluate(valid_loader)

       valid_score = valid_result[self.valid_metric]
       return valid_score, valid_result
   

    
    @torch.no_grad()
    def evaluate(self, vali_loader, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """

        self.model.eval()
        all_outputs = [] 
        all_gt = []
  
        for batch_idx, (batch_x,batch_prompt,batch_y) in tqdm(enumerate(vali_loader),desc="Evaluating"):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs_np = self.model(batch_x,batch_prompt,flag='test').detach().cpu().numpy()
            targets_np = batch_y.detach().cpu().numpy()
            all_outputs.append(outputs_np)
            all_gt.append(targets_np)
            

                

        
        self.model.train()
        return self.evaluator.evaluate(all_outputs,all_gt)


      

    def _plot_progress(self):
        """训练过程可视化"""
        plt.figure(figsize=(12, 5))
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Training Progress')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / 'training_curve.png')
        plt.close()

    def fit(self, train_loader, valid_loader, test_loader=None,saved = False,verbose = True):
        """执行完整训练流程"""

        for epoch_idx in range(self.start_epoch, self.epochs):
                # train
                self.model.train()
                training_start_time = time()
                loss_func = nn.MSELoss() 
                
                train_loss, _ = self.train_epoch(train_loader, epoch_idx,loss_func=loss_func)

                self.lr_scheduler.step()

                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
             
                if verbose:
                    self.logger.info(train_loss_output)
                if (epoch_idx + 1) % self.eval_step == 0:
                    valid_start_time = time()
                    valid_score, valid_result = self._valid_epoch(valid_loader)
                    self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                        valid_score, self.best_valid_score, self.cur_step,
                        max_step=self.stopping_step)
                    valid_end_time = time()
                    valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                        (epoch_idx, valid_end_time - valid_start_time, valid_score)
                    valid_result_output = 'valid result: \n' + dict2str(valid_result)
                    # test
                    _, test_result = self._valid_epoch(test_loader)
                    if verbose:
                        self.logger.info(valid_score_output)
                        self.logger.info(valid_result_output)
                        self.logger.info('test result: \n' + dict2str(test_result))
                    if update_flag:
                     
                        
                        update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                        if verbose:
                            self.logger.info(update_output)
                        self.best_valid_result = valid_result
                        self.best_test_upon_valid = test_result

                    if stop_flag:
                        stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                    (epoch_idx - self.cur_step * self.eval_step)
                        if verbose:
                            self.logger.info(stop_output)
                        break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'