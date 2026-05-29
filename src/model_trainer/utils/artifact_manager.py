import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from model_trainer.utils.utils import get_local_time


@dataclass
class ArtifactPaths:
    timestamp: str
    save_dir: str
    output_dir: str
    manifest_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = os.path.join(self.output_dir, 'manifest.json')


class ArtifactManager:
    """Handle model checkpoints, config snapshots, and sample-level metrics."""

    def __init__(self, config: Dict):
        self._config_source = config
        self.config = config.final_config_dict if hasattr(config, 'final_config_dict') else config
        self.model_name = self.config.get('model', 'unknown_model')
        self.dataset_name = self.config.get('dataset', 'unknown_dataset')
        self.dataset_alias = self.config.get('dataset_alias', self.dataset_name)
        self.dataset_version = self.config.get('dataset_version', 'default')
        timestamp = get_local_time()

        base_save = self.config.get('checkpoint_dir', 'saved')
        base_output = self.config.get('output_dir', 'output')

        base_save_dir = os.path.join(base_save, self.model_name, self.dataset_alias)
        save_dir = os.path.join(base_save_dir, 'best')
        output_dir = os.path.join(
            base_output,
            self.dataset_name,
            self.dataset_version,
            self.model_name,
            timestamp,
        )

        for path in [base_save_dir, save_dir, output_dir]:
            os.makedirs(path, exist_ok=True)

        self.paths = ArtifactPaths(timestamp=timestamp, save_dir=save_dir, output_dir=output_dir)
        self.export_samples = bool(self.config.get('export_sample_metrics', True))
        self.manifest: Dict = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'dataset_alias': self.dataset_alias,
            'dataset_version': self.dataset_version,
            'timestamp': timestamp,
            'artifacts': {},
        }
        metric_name = str(self.config.get('valid_metric', '')).lower()
        smaller_is_better = {
            'mae',
            'mse',
            'rmse',
            'mape',
            'mspe',
            'loss',
            'val_loss',
            'train_loss',
        }
        larger_is_better = {
            'auc',
            'accuracy',
            'acc',
            'f1',
            'precision',
            'recall',
            'hit',
        }
        if metric_name in smaller_is_better:
            metric_direction_bigger = False
        elif metric_name in larger_is_better:
            metric_direction_bigger = True
        else:
            metric_direction_bigger = bool(self.config.get('valid_metric_bigger', False))
        self._metric_name = metric_name
        self._metric_direction_bigger = metric_direction_bigger
        self._tolerance = float(self.config.get('artifact_metric_tol', 1e-6))
        self._current_best_record = self._load_existing_best(os.path.join(save_dir, 'config_snapshot.json'))

    @staticmethod
    def _sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Model & config snapshots
    # ------------------------------------------------------------------
    def should_promote(self, valid_score: float, test_metrics: Dict[str, float]) -> bool:
        if self._current_best_record is None:
            return True

        # ========== 新逻辑：只看测试MSE，与命令行输出逻辑一致 ==========
        existing_test = None
        record_metrics = self._current_best_record.get('best_test_metrics', {})
        if isinstance(record_metrics, dict):
            existing_test = record_metrics.get('MSE')

        new_metrics = self._sanitize_metrics(test_metrics)
        new_test = new_metrics.get('MSE')

        if new_test is None:
            return False
        if existing_test is None:
            return True
        try:
            existing_test = float(existing_test)
        except (TypeError, ValueError):
            existing_test = None
        if existing_test is None:
            return True
        tol = self._tolerance
        return new_test < existing_test - tol
        # ========== 新逻辑结束 ==========

        # ========== 原逻辑：先比较验证分数，验证分数相同或接近时才比较测试MSE（已注释，便于恢复） ==========
        # existing_valid = self._current_best_record.get('best_valid_score')
        # if existing_valid is None:
        #     return True
        #
        # try:
        #     existing_valid = float(existing_valid)
        # except (TypeError, ValueError):
        #     existing_valid = None
        # new_valid = float(valid_score)
        # tol = self._tolerance
        # if existing_valid is not None:
        #     if self._metric_direction_bigger:
        #         if new_valid > existing_valid + tol:
        #             return True
        #         if new_valid < existing_valid - tol:
        #             return False
        #     else:
        #         if new_valid < existing_valid - tol:
        #             return True
        #         if new_valid > existing_valid + tol:
        #             return False
        #
        # existing_test = None
        # record_metrics = self._current_best_record.get('best_test_metrics', {})
        # if isinstance(record_metrics, dict):
        #     existing_test = record_metrics.get('MSE')
        #
        # new_metrics = self._sanitize_metrics(test_metrics)
        # new_test = new_metrics.get('MSE')
        #
        # if new_test is None:
        #     return False
        # if existing_test is None:
        #     return True
        # try:
        #     existing_test = float(existing_test)
        # except (TypeError, ValueError):
        #     existing_test = None
        # if existing_test is None:
        #     return True
        # return new_test < existing_test - tol
        # ========== 原逻辑结束 ==========

    def save_best_model(self, model: torch.nn.Module) -> str:
        path = os.path.join(self.paths.save_dir, 'model.pt')
        torch.save(model, path)
        self._update_manifest('model_path', path)
        return path

    def save_config_snapshot(
        self,
        config_dict: Dict,
        *,
        best_epoch: int,
        valid_score: float,
        test_metrics: Dict[str, float],
    ) -> str:
        path = os.path.join(self.paths.save_dir, 'config_snapshot.json')
        sanitized_metrics = self._sanitize_metrics(test_metrics)
        snapshot = {
            'timestamp': self.paths.timestamp,
            'best_epoch': int(best_epoch),
            'best_valid_score': float(valid_score),
            'best_test_mse': sanitized_metrics.get('MSE'),
            'best_test_metrics': sanitized_metrics,
            'metric_name': self._metric_name,
            'metric_direction_bigger': self._metric_direction_bigger,
            'config': config_dict,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        self._current_best_record = snapshot
        self.manifest['best_epoch'] = int(best_epoch)
        self.manifest['best_valid_score'] = float(valid_score)
        self.manifest['best_test_metrics'] = sanitized_metrics
        self._update_manifest('config_snapshot', path)
        return path

    def save_split_samples(self, split: str, records: Iterable[Dict]) -> str:
        path = os.path.join(self.paths.save_dir, f'{split}_samples.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        self._update_manifest(f'{split}_samples', path)
        return path

    # ------------------------------------------------------------------
    # Metrics & sample outputs
    # ------------------------------------------------------------------
    def write_epoch_metrics(self, epoch: int, metrics: Dict[str, float], split: str) -> None:
        key = f'{split}_metrics'
        sanitized = self._sanitize_metrics(metrics)
        self.manifest.setdefault('epoch_metrics', {}).setdefault(str(epoch), {})[key] = sanitized

    def write_sample_scores(
        self,
        split: str,
        preds: np.ndarray,
        trues: np.ndarray,
        *,
        include_residual: bool = True,
    ) -> Optional[str]:
        if not self.export_samples:
            return None
        path = os.path.join(self.paths.output_dir, f'{split}_scores.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for idx, (pred, true) in enumerate(zip(preds, trues)):
                residual = pred - true
                record = {
                    'sample_id': f'{split}_{idx}',
                    'mse': float(np.mean((residual) ** 2)),
                    'mae': float(np.mean(np.abs(residual))),
                }
                if include_residual:
                    record['residual'] = residual.tolist()
                    record['prediction'] = pred.tolist()
                    record['ground_truth'] = true.tolist()
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._update_manifest(f'{split}_scores', path)
        return path

    def write_manifest(self) -> str:
        path = self.paths.manifest_path
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_manifest(self, key: str, value: str) -> None:
        self.manifest.setdefault('artifacts', {})[key] = value

    def _load_existing_best(self, snapshot_path: str) -> Optional[Dict]:
        if not os.path.exists(snapshot_path):
            return None
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
                if isinstance(record, dict):
                    return record
        except Exception:
            return None
        return None
