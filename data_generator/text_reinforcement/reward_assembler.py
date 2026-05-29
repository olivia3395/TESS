from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    reward_clip: float = 1.0
    min_reward: float = -0.2


class RewardAssembler:
    def __init__(
        self,
        *,
        dataset_root: Path,
        base_version: str,
        turn_id: int,
        variant_versions: Mapping[str, str],
        scores_root: Path,
        reward_config: RewardConfig,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.base_version = base_version
        self.turn_id = turn_id
        self.variant_versions = dict(variant_versions)
        self.scores_root = Path(scores_root)
        self.reward_config = reward_config

    def run(self) -> Dict[str, float]:
        base_scores = self._load_scores(self.scores_root / self.base_version / 'train_scores.jsonl')
        variant_scores = {
            variant: self._load_scores(self.scores_root / version / 'train_scores.jsonl')
            for variant, version in self.variant_versions.items()
        }
        base_records = self._load_dataset_records(self.base_version)
        variant_records = {
            variant: self._load_dataset_records(version)
            for variant, version in self.variant_versions.items()
        }

        output_dir = self.dataset_root / f"trl_turn{self.turn_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'dpo_pairs.jsonl'

        chosen = 0
        skipped = 0
        with output_path.open('w', encoding='utf-8') as f:
            for sample_id, ref_score in base_scores.items():
                base_record = base_records.get(sample_id)
                if not base_record:
                    skipped += 1
                    continue

                candidates = []
                base_news = base_record.get('news', '').strip()
                if base_news:
                    candidates.append({
                        'variant': 'base',
                        'news': base_news,
                        'mse': ref_score['mse'],
                    })

                for variant, scores in variant_scores.items():
                    if sample_id not in scores:
                        continue
                    variant_record = variant_records.get(variant, {}).get(sample_id)
                    if not variant_record:
                        continue
                    news_text = str(variant_record.get('news', '')).strip()
                    if not news_text:
                        continue
                    candidates.append({
                        'variant': variant,
                        'news': news_text,
                        'mse': scores[sample_id]['mse'],
                    })

                if len(candidates) < 2:
                    skipped += 1
                    continue

                best_entry = min(candidates, key=lambda item: item['mse'])
                worst_entry = max(candidates, key=lambda item: item['mse'])
                if best_entry['mse'] >= worst_entry['mse']:
                    skipped += 1
                    continue

                raw_reward = worst_entry['mse'] - best_entry['mse']
                reward = max(min(raw_reward, self.reward_config.reward_clip), -self.reward_config.reward_clip)
                if reward < self.reward_config.min_reward:
                    skipped += 1
                    continue

                metadata = {
                    'turn_id': self.turn_id,
                    'preferred_variant': best_entry['variant'],
                    'rejected_variant': worst_entry['variant'],
                    'positive_mse': best_entry['mse'],
                    'negative_mse': worst_entry['mse'],
                    'base_mse': ref_score['mse'],
                }
                for entry in candidates:
                    metadata[f"{entry['variant']}_mse"] = entry['mse']

                record = {
                    'sample_id': sample_id,
                    'historical_data': base_record['historical_data'],
                    'news_positive': best_entry['news'],
                    'news_negative': worst_entry['news'],
                    'ground_truth': base_record['ground_truth'],
                    'reward': reward,
                    'reward_raw': raw_reward,
                    'metadata': metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                chosen += 1
        logger.info(
            "RewardAssembler: wrote %d records to %s (skipped=%d)",
            chosen,
            output_path,
            skipped,
        )
        total = chosen + skipped
        return {
            'total_samples': float(total),
            'selected_samples': float(chosen),
            'skip_ratio': float(skipped / total) if total else 0.0,
        }

    def _load_scores(self, path: Path) -> Dict[str, Dict[str, float]]:
        if not path.exists():
            raise FileNotFoundError(f'Score file not found: {path}')
        scores: Dict[str, Dict[str, float]] = {}
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                scores[record['sample_id']] = record
        return scores

    def _load_dataset_records(self, version: str) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        dataset_dir = self.dataset_root / version
        path = dataset_dir / 'train.json'
        if not path.exists():
            raise FileNotFoundError(f'Dataset split missing: {path}')
        with path.open('r', encoding='utf-8') as f:
            entries: List[Dict[str, str]] = json.load(f)
        for idx, item in enumerate(entries):
            sample_id = item.get('source_sample_id') or item.get('sample_id') or f'train-{idx:05d}'
            result[sample_id] = item
        return result
