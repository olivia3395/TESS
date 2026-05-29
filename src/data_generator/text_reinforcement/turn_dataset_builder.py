from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_trainer.common import dataloader as dataloader_module
from model_trainer.utils.configurator import Config
from model_trainer.utils.dataset_registry import DatasetRegistry
from model_trainer.utils.embedding_checker import ensure_embeddings
from model_trainer.utils.utils import get_model

logger = logging.getLogger(__name__)


@dataclass
class ScoreRecord:
    sample_id: str
    mse: float
    prediction: List[float]
    ground_truth: List[float]


class TurnDatasetBuilder:
    def __init__(
        self,
        *,
        dataset_alias: str,
        dataset_name: str,
        dataset_version: str,
        model_name: str,
        checkpoint_path: Path,
        batch_size: int = 32,
        device: Optional[str] = None,
        auto_generate_embedding: bool = True,
        metrics_in_original_scale: bool = False,
    ) -> None:
        self.dataset_alias = dataset_alias
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.auto_generate_embedding = auto_generate_embedding
        self.metrics_in_original_scale = metrics_in_original_scale

    def run(self, output_root: Path) -> Dict[str, Dict[str, float]]:
        logger.info(
            "TurnDatasetBuilder: evaluating model %s on alias %s", self.model_name, self.dataset_alias
        )
        DatasetRegistry._load_index.cache_clear()  # ensure latest alias available

        config_dict: Dict = {
            'dataset_alias': self.dataset_alias,
            'dataset_version': self.dataset_version,
            'dataset': self.dataset_name,
            'model': self.model_name,
            'batch_size': self.batch_size,
            'use_gpu': self.device.type == 'cuda',
            'gpu_id': self.device.index or 0,
            'auto_generate_embedding': self.auto_generate_embedding,
            'export_sample_metrics': False,
            'legacy_loader': False,
            'req_training': False,
        }
        config = Config(config_dict, self.model_name, self.dataset_name)
        config['req_training'] = False
        config['export_sample_metrics'] = False
        config['auto_generate_embedding'] = self.auto_generate_embedding
        config['use_gpu'] = self.device.type == 'cuda'
        config['gpu_id'] = self.device.index or 0
        config['batch_size'] = self.batch_size

        ensure_embeddings(config.final_config_dict, logger=logger)

        dataset_cls = dataloader_module.data_dict[self.dataset_name]
        train_dataset = dataset_cls(config, flag='train')
        scaler = train_dataset.get_scaler()
        vali_dataset = dataset_cls(config, flag='vali', scaler=scaler)
        test_dataset = dataset_cls(config, flag='test', scaler=scaler)

        loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=dataloader_module.custom_collate_fn,
            ),
            'vali': DataLoader(
                vali_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=dataloader_module.custom_collate_fn,
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=dataloader_module.custom_collate_fn,
            ),
        }

        model_cls = get_model(self.model_name)
        model = model_cls(config).to(self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, torch.nn.Module):
            model = state.to(self.device)
        else:
            model.load_state_dict(state)
        model.eval()

        split_metrics: Dict[str, Dict[str, float]] = {}
        turn_output_dir = Path(output_root) / self.dataset_name / self.dataset_version
        turn_output_dir.mkdir(parents=True, exist_ok=True)

        for split, loader in loaders.items():
            dataset_obj = loader.dataset
            scores, metrics = self._score_split(model, loader, dataset_obj, split)
            split_metrics[split] = metrics
            out_path = turn_output_dir / f"{split}_scores.jsonl"
            self._write_scores(out_path, scores)
            logger.info(
                "TurnDatasetBuilder: wrote %s scores (%d samples, MSE=%.6f)",
                split,
                len(scores),
                metrics['MSE'],
            )

        return split_metrics

    def _score_split(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        dataset_obj,
        split: str,
    ) -> Tuple[List[ScoreRecord], Dict[str, float]]:
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                prepared = self._prepare_batch(batch)
                outputs = self._forward_model(model, prepared)
                preds.append(outputs.cpu().numpy())
                trues.append(prepared['y'].cpu().numpy())

        preds_arr = np.concatenate(preds, axis=0)
        trues_arr = np.concatenate(trues, axis=0)
        if self.metrics_in_original_scale and hasattr(dataset_obj, 'inverse_transform'):
            preds_arr = self._inverse_batch(dataset_obj, preds_arr)
            trues_arr = self._inverse_batch(dataset_obj, trues_arr)

        mse = float(np.mean((preds_arr - trues_arr) ** 2))
        mae = float(np.mean(np.abs(preds_arr - trues_arr)))
        records: List[ScoreRecord] = []
        for idx, (pred, true) in enumerate(zip(preds_arr, trues_arr)):
            meta = dataset_obj.data[idx] if hasattr(dataset_obj, 'data') else {}
            sample_id = meta.get('source_sample_id') or meta.get('sample_id') or f"{split}-{idx:05d}"
            mse_i = float(np.mean((pred - true) ** 2))
            records.append(
                ScoreRecord(
                    sample_id=sample_id,
                    mse=mse_i,
                    prediction=pred.tolist(),
                    ground_truth=true.tolist(),
                )
            )
        metrics = {
            'MSE': mse,
            'MAE': mae,
        }
        return records, metrics

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = {}
        for key in ('x', 'y', 'meta_tensor', 'news_embed'):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        prepared['news_text'] = batch.get('news_text')
        prepared['batch_size'] = prepared['x'].size(0)
        return prepared

    def _forward_model(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_name = model.__class__.__name__
        batch_x = batch['x']
        if model_name == 'TimeLLM':
            news_inputs = batch['news_text'] or [""] * batch['batch_size']
            outputs = model(batch_x, news=news_inputs, flag='test')
        elif model_name in {'CMA', 'MultiModal_Baseline'}:
            news_feat = batch.get('news_embed')
            if news_feat is None:
                news_feat = batch.get('meta_tensor')
            if news_feat is None:
                raise ValueError(f"Model {model_name} requires news embeddings but batch lacks them")
            if 'flag' in model.forward.__code__.co_varnames:
                outputs = model(batch_x, news_feat, flag='test')
            else:
                outputs = model(batch_x, news_feat)
        else:
            if 'flag' in model.forward.__code__.co_varnames:
                outputs = model(batch_x, flag='test')
            else:
                outputs = model(batch_x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs.float()

    def _inverse_batch(self, dataset_obj, arr: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(arr).float()
        restored = dataset_obj.inverse_transform(tensor)
        if isinstance(restored, torch.Tensor):
            restored = restored.cpu().numpy()
        return restored

    def _write_scores(self, path: Path, records: Iterable[ScoreRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            'sample_id': record.sample_id,
                            'mse': record.mse,
                            'prediction': record.prediction,
                            'ground_truth': record.ground_truth,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
