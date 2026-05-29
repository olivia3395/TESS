# coding: utf-8
import datetime
import os
import numpy as np
import torch
from typing import List, Dict, Union
import json



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


metrics_dict = {
    'mae': MAE,
    'mse': MSE,
    'rmse': RMSE,
    'mape': MAPE,
    'mspe': MSPE,
    'rse': RSE,
    'corr': CORR
}

class TemporalEvaluator:
    """Evaluator for time series forecasting tasks.
    Supported metrics: MAE, RMSE, MAPE (%), R2
    """
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = self._validate_metrics(config['metrics'])

        
    def collect(self, 
                batch_predictions: torch.Tensor, 
                batch_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Collect batch results with shape (batch_size, forecast_horizon)"""
        return {
            'predictions': batch_predictions.detach().cpu(),
            'targets': batch_targets.detach().cpu()
        }

    def evaluate(self,pred,target):
        """Calculate metrics on all collected data"""
        # Concatenate all batches
        # import ipdb; ipdb.set_trace()
        preds = np.array(pred)
        targets = np.array(target)
        preds_full = np.concatenate(preds,axis=0)
        targets_full = np.concatenate(targets,axis=0)
        
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])
        # Calculate metrics
        results = {}
        for metric in self.metrics:
            metric_func = metrics_dict[metric.lower()]
            value = metric_func(preds_full,targets_full)
            results[metric.upper()] = float(value)  # Convert to Python float
            
        return results 

    def _validate_metrics(self, metrics: Union[str, List[str]]) -> List[str]:
        """Validate input metrics"""
        if isinstance(metrics, str):
            metrics = [metrics]
        elif not isinstance(metrics, list):
            raise TypeError("Metrics must be str or list[str]")
            
        valid_metrics = []
        for m in metrics:
            m_lower = m.lower()
            if m_lower not in metrics_dict:
                raise ValueError(f"Unsupported metric: {m}. Choose from {list(metrics_dict.keys())}")
            valid_metrics.append(m_lower)
            
        return valid_metrics

    def _save_results(self, 
                     predictions: np.ndarray,
                     targets: np.ndarray,
                     metrics: Dict[str, float]) -> None:
        """Save predictions and metrics to files"""

        
        # Save numerical results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        with open(os.path.join(self.save_dir, f'metrics_{timestamp}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def __str__(self) -> str:
        return f"TemporalEvaluator(metrics={[m.upper() for m in self.metrics]})"
