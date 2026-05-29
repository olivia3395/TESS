from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from model_trainer.utils.dataset_registry import DatasetRegistry

from ..utils.llm_client import LLMClient, LLMResult
from .augment_orchestrator import AugmentMetrics, AugmentOrchestrator
from .async_executor import AsyncBatchExecutor
from .prompt_builder import load_prompt_template
from .reward_assembler import RewardAssembler, RewardConfig
from .turn_dataset_builder import TurnDatasetBuilder

logger = logging.getLogger("trl_pipeline")


@dataclass
class PipelineConfig:
    batch_size: int = 32
    initial_concurrency: int = 120
    max_concurrency: int = 500
    min_concurrency: int = 64
    latency_increase_threshold: float = 1.0
    latency_decrease_threshold: float = 2.0
    perf_window: int = 50
    max_retries: int = 3
    retry_backoff: float = 0.75
    retry_multiplier: float = 1.7
    reward_clip: float = 1.0
    min_reward: float = -0.2
    max_tokens: int = 320
    resume_failures: bool = True
    progress_log_interval: int = 1000
    metrics_in_original_scale: bool = False

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        with path.open('r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        data = {**asdict(cls()), **raw}
        return cls(**data)


class DummyLLMClient:
    """Dry-run LLM client for testing."""

    async def achat(self, *, user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 320, **_: Dict) -> LLMResult:  # type: ignore[override]
        await asyncio.sleep(0.01)
        content = f"[DUMMY OUTPUT] {user_prompt[:200]}"
        return LLMResult(content=content, prompt_tokens=len(user_prompt.split()), completion_tokens=min(max_tokens, 120), cost=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRL augmentation turn pipeline")
    parser.add_argument('--base-alias', required=True, help='Dataset alias, e.g. FNSPID/ver_camf')
    parser.add_argument('--turn', type=int, required=True, help='Turn identifier (integer)')
    parser.add_argument('--variants', nargs='+', default=['aug1', 'aug2'], help='Augmentation variant keys')
    parser.add_argument('--checkpoint', required=True, help='Path to frozen model checkpoint (torch loadable)')
    parser.add_argument('--model', required=True, help='Model name to instantiate (e.g. MultiModal_Baseline)')
    parser.add_argument('--prompt-config', default='configs/text_reinforcement/augment_prompt.yaml')
    parser.add_argument('--pipeline-config', default='configs/text_reinforcement/pipeline.yaml')
    parser.add_argument('--dataset-root', help='Dataset root override; defaults to registry root for base alias')
    parser.add_argument('--output-root', default='output', help='Root directory for score outputs (default: output)')
    parser.add_argument('--scores-root', help='Root directory for reading reference scores (default: <output-root>/<dataset_name>)')
    parser.add_argument('--max-concurrency', type=int, help='Override max concurrency limit')
    parser.add_argument('--dry-run', action='store_true', help='Use dummy LLM client instead of hitting remote service')
    parser.add_argument('--no-embedding', action='store_true', help='Skip automatic embedding generation for augmented datasets')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='[%(levelname)s] %(name)s: %(message)s')


async def _run_async(args: argparse.Namespace) -> Dict[str, Dict]:
    pipeline_cfg = PipelineConfig.load(Path(args.pipeline_config))
    if args.max_concurrency:
        pipeline_cfg.max_concurrency = args.max_concurrency

    registry_info = DatasetRegistry.get(args.base_alias)
    dataset_root = Path(args.dataset_root or registry_info.get('root') or '.')
    dataset_name = registry_info.get('dataset_name') or args.base_alias.split('/')[0]
    base_version = registry_info.get('version') or args.base_alias.split('/')[-1]

    prompt_config = Path(args.prompt_config)
    load_prompt_template(prompt_config)  # warm cache to validate early

    executor = AsyncBatchExecutor(
        initial_concurrency=pipeline_cfg.initial_concurrency,
        max_concurrency=pipeline_cfg.max_concurrency,
        min_concurrency=pipeline_cfg.min_concurrency,
        latency_increase_threshold=pipeline_cfg.latency_increase_threshold,
        latency_decrease_threshold=pipeline_cfg.latency_decrease_threshold,
        perf_window=pipeline_cfg.perf_window,
        max_retries=pipeline_cfg.max_retries,
        retry_backoff=pipeline_cfg.retry_backoff,
        retry_multiplier=pipeline_cfg.retry_multiplier,
    )

    if args.dry_run:
        llm_client = DummyLLMClient()
    else:
        llm_client = LLMClient(max_retries=0)

    orchestrator = AugmentOrchestrator(
        dataset_root=dataset_root,
        base_version=base_version,
        turn_id=args.turn,
        prompt_config=prompt_config,
        executor=executor,
        llm_client=llm_client,  # type: ignore[arg-type]
        batch_size=pipeline_cfg.batch_size,
        max_tokens=pipeline_cfg.max_tokens,
        resume_failures=pipeline_cfg.resume_failures,
        progress_log_interval=pipeline_cfg.progress_log_interval,
    )

    augment_metrics = await orchestrator.run(args.variants)
    _register_new_aliases(dataset_root, dataset_name, base_version, args.turn, args.variants)

    output_root = Path(args.output_root)
    dataset_output_root = output_root / dataset_name
    turn_variants = {variant: f"ver_turn{args.turn}_{variant}" for variant in args.variants}
    split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant, version in turn_variants.items():
        builder = TurnDatasetBuilder(
            dataset_alias=f"{dataset_name}/{version}",
            dataset_name=dataset_name,
            dataset_version=version,
            model_name=args.model,
            checkpoint_path=Path(args.checkpoint),
            batch_size=pipeline_cfg.batch_size,
            auto_generate_embedding=not args.no_embedding,
            metrics_in_original_scale=pipeline_cfg.metrics_in_original_scale,
        )
        variant_metrics = builder.run(output_root)
        split_metrics[variant] = variant_metrics

    scores_root = Path(args.scores_root) if args.scores_root else dataset_output_root
    reward_assembler = RewardAssembler(
        dataset_root=dataset_root,
        base_version=base_version,
        turn_id=args.turn,
        variant_versions=turn_variants,
        scores_root=scores_root,
        reward_config=RewardConfig(
            reward_clip=pipeline_cfg.reward_clip,
            min_reward=pipeline_cfg.min_reward,
        ),
    )
    reward_stats = reward_assembler.run()

    remaining_failures = sum(metric.failed_samples for metric in augment_metrics.values())
    metrics_summary = {
        'turn': args.turn,
        'dataset': dataset_name,
        'variants': turn_variants,
        'augment': {variant: asdict(metric) for variant, metric in augment_metrics.items()},
        'scores': split_metrics,
        'reward': reward_stats,
        'executor': {
            'target_concurrency': executor.target_concurrency,
            'max_observed_concurrency': executor.max_observed_concurrency,
            'rate_limit_events': executor.rate_limit_events,
        },
        'remaining_failures': remaining_failures,
        'output_root': str(dataset_output_root),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

    repo_root = Path(__file__).resolve().parents[3]
    metrics_dir = repo_root / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f'trl_turn{args.turn}.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    meta_dir = dataset_output_root / f'ver_turn{args.turn}_meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    legacy_metrics_path = meta_dir / 'metrics.json'
    with legacy_metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    logger.info('Pipeline complete. Metrics saved to %s', metrics_path)
    return metrics_summary


def _register_new_aliases(dataset_root: Path, dataset_name: str, base_version: str, turn_id: int, variants: List[str]) -> None:
    index_path = Path(__file__).resolve().parents[2] / 'model_trainer' / 'configs' / 'dataset' / 'index.yaml'
    with index_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    aliases = data.setdefault('aliases', {})
    repo_root = Path(__file__).resolve().parents[3]
    for variant in variants:
        version = f'ver_turn{turn_id}_{variant}'
        alias_key = f'{dataset_name}/{version}'
        root_rel = os.path.relpath(dataset_root, repo_root)
        aliases[alias_key] = {
            'dataset_name': dataset_name,
            'version': version,
            'root': root_rel.replace('\\', '/'),
            'splits': {
                'train': f'{version}/train.json',
                'vali': f'{version}/vali.json',
                'test': f'{version}/test.json',
            },
            'embeddings': {
                'news': {
                    'path': f'{version}/embedding_qwen/all_embeddings.pt',
                    'splits': {
                        'train': 'train_news',
                        'vali': 'vali_news',
                        'test': 'test_news',
                    },
                }
            },
        }
        embed_dir = dataset_root / version / 'embedding_qwen'
        embed_dir.mkdir(parents=True, exist_ok=True)
    with index_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)
    DatasetRegistry._load_index.cache_clear()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    start_time = time.perf_counter()
    metrics = asyncio.run(_run_async(args))
    duration = time.perf_counter() - start_time
    logger.info('Total pipeline duration %.2fs', duration)
    if not args.dry_run:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
