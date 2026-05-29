#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from sentence_transformers import SentenceTransformer

from model_trainer.utils.dataset_registry import DatasetRegistry, DatasetRegistryError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen embeddings for a dataset alias")
    parser.add_argument('--alias', required=True, help='Dataset alias, e.g. FNSPID/ver_camf')
    parser.add_argument('--dataset-version', help='Optional dataset version override')
    parser.add_argument('--model-path', default='/ssd/hf_home/models/Qwen3-Embedding-8B', help='Local embedding model path')
    parser.add_argument('--batch-size', type=int, default=32, help='Encoding batch size')
    parser.add_argument('--device', default=None, help='Device for embedding model (e.g. cpu, cuda)')
    parser.add_argument(
        '--devices',
        default=None,
        help='Comma separated device list (e.g. cuda:0,cuda:1). Overrides --device.',
    )
    return parser.parse_args()


def load_registry(alias: str, dataset_version: str = None) -> Dict:
    overrides = None
    if dataset_version:
        overrides = {'version': dataset_version}
    try:
        return DatasetRegistry.get(alias, overrides=overrides)
    except DatasetRegistryError as exc:
        raise RuntimeError(f'Failed to resolve dataset alias {alias}: {exc}') from exc


def load_model(model_path: str, device: str = None) -> SentenceTransformer:
    if device:
        return SentenceTransformer(model_path, device=device)
    return SentenceTransformer(model_path)


def read_split_texts(dataset_root: str, relative_path: str) -> List[str]:
    path = os.path.abspath(os.path.join(dataset_root, relative_path.lstrip('/')))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Split file not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    texts = []
    for item in records:
        if isinstance(item, dict):
            text = item.get('news') or item.get('prompt') or ''
        else:
            text = str(item)
        texts.append(text)
    return texts


def _normalise_device_ordinals(devices: List[Optional[str]]) -> List[Optional[str]]:
    if not torch.cuda.is_available():
        return devices

    available = torch.cuda.device_count()
    if available == 0:
        return devices

    env_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    env_list: List[str] = []
    if env_visible:
        env_list = [entry.strip() for entry in env_visible.split(',') if entry.strip()]

    normalised: List[Optional[str]] = []
    for idx, device in enumerate(devices):
        if device is None or not device.startswith('cuda'):
            normalised.append(device)
            continue
        try:
            _, ordinal_str = device.split(':', 1)
            ordinal = int(ordinal_str)
        except ValueError:
            normalised.append(device)
            continue
        if ordinal < available:
            normalised.append(device)
            continue
        if env_list and idx < available:
            normalised.append(f'cuda:{idx}')
        else:
            raise ValueError(
                f'Device {device} is not available (visible CUDA count={available}). '
                'Consider setting CUDA_VISIBLE_DEVICES or adjust --devices.',
            )
    return normalised


def _resolve_devices(single_device: Optional[str], multi_devices: Optional[str]) -> List[Optional[str]]:
    if multi_devices and single_device:
        raise ValueError('Cannot specify both --device and --devices')
    if multi_devices:
        devices = [entry.strip() for entry in multi_devices.split(',') if entry.strip()]
        if not devices:
            raise ValueError('No valid devices provided in --devices')
        return _normalise_device_ordinals(devices)
    if single_device:
        entries = [entry.strip() for entry in single_device.split(',') if entry.strip()]
        if not entries:
            return [None]
        return _normalise_device_ordinals(entries)
    return [None]


def _split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
    if parts <= 0:
        return [(0, total)]
    base = total // parts
    remainder = total % parts
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        extra = 1 if idx < remainder else 0
        end = min(total, start + base + extra)
        ranges.append((start, end))
        start = end
    return ranges


def _encode_worker(task: Tuple[Optional[str], str, int, List[str]]) -> np.ndarray:
    device, model_path, batch_size, chunk = task
    model = load_model(model_path, device)
    return model.encode(
        chunk,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )


def encode_sentences(
    sentences: Sequence[str],
    model_path: str,
    batch_size: int,
    devices: Sequence[Optional[str]],
    cached_model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    sentences_list = list(sentences)
    if not sentences_list:
        raise RuntimeError('No sentences provided for embedding generation')

    if len(devices) == 1:
        model = cached_model or load_model(model_path, devices[0])
        return model.encode(
            sentences_list,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False,
        )

    # Multi-device path
    ranges = _split_ranges(len(sentences_list), len(devices))
    tasks: List[Tuple[Optional[str], str, int, List[str]]] = []
    for device, (start, end) in zip(devices, ranges):
        chunk = sentences_list[start:end]
        if not chunk:
            continue
        tasks.append((device, model_path, batch_size, chunk))

    if not tasks:
        raise RuntimeError('No work scheduled for multi-device encoding')

    ctx = mp.get_context('spawn')
    with ctx.Pool(len(tasks)) as pool:
        results = pool.map(_encode_worker, tasks)

    return np.concatenate(results, axis=0)


def main() -> None:
    args = parse_args()
    registry_info = load_registry(args.alias, args.dataset_version)

    dataset_root = registry_info.get('root')
    if not dataset_root:
        raise RuntimeError('Dataset registry entry missing root path')

    embeddings = registry_info.get('embeddings', {})
    news_spec = embeddings.get('news')
    if not news_spec:
        raise RuntimeError('Dataset registry entry missing news embedding specification')

    embed_rel_path = news_spec.get('path')
    if not embed_rel_path:
        raise RuntimeError('Embedding path not specified in registry')

    splits = registry_info.get('splits', {})
    embed_split_keys = news_spec.get('splits', {})

    devices = _resolve_devices(args.device, args.devices)
    cached_model: Optional[SentenceTransformer] = None
    if len(devices) == 1:
        cached_model = load_model(args.model_path, devices[0])

    output: Dict[str, torch.Tensor] = {}
    device_desc = ', '.join(device or 'auto' for device in devices)
    for split, rel_file in splits.items():
        if split not in embed_split_keys:
            continue
        sentences = read_split_texts(dataset_root, rel_file)
        if not sentences:
            raise RuntimeError(f'No records found for split {split} at {rel_file}')
        print(f'Encoding split {split} with devices: {device_desc}')
        embeddings_np = encode_sentences(
            sentences,
            model_path=args.model_path,
            batch_size=args.batch_size,
            devices=devices,
            cached_model=cached_model,
        )
        output_key = embed_split_keys[split]
        output[output_key] = torch.from_numpy(embeddings_np).float()
        print(f'Generated embeddings for {split}: shape={embeddings_np.shape}')

    embed_abs_path = os.path.abspath(os.path.join(dataset_root, embed_rel_path.lstrip('/')))
    os.makedirs(os.path.dirname(embed_abs_path), exist_ok=True)
    torch.save(output, embed_abs_path)
    print(f'Embeddings saved to {embed_abs_path}')


if __name__ == '__main__':
    main()
