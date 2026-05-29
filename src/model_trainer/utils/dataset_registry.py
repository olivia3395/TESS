import copy
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml


class DatasetRegistryError(RuntimeError):
    """Raised when dataset alias resolution fails."""


class DatasetRegistry:
    """Resolve dataset aliases into concrete paths and metadata."""

    _INDEX_REL_PATH = os.path.join('configs', 'dataset', 'index.yaml')

    @classmethod
    def _index_path(cls) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, cls._INDEX_REL_PATH)

    @classmethod
    @lru_cache(maxsize=1)
    def _load_index(cls) -> Dict[str, Any]:
        path = cls._index_path()
        if not os.path.isfile(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data.get('aliases', {})

    @classmethod
    def get(cls, alias: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not alias:
            raise DatasetRegistryError('dataset alias is required')
        index = cls._load_index()
        if alias not in index:
            raise DatasetRegistryError(f'unknown dataset alias: {alias}')

        entry = copy.deepcopy(index[alias])
        if overrides:
            entry = _deep_merge(entry, overrides)

        entry['alias'] = alias
        entry['root'] = _resolve_path(entry.get('root'))

        splits = entry.get('splits', {})
        entry['splits'] = {
            name: _normalise_relative_path(path) for name, path in splits.items()
        }

        embeddings = entry.get('embeddings', {})
        normalised_embeddings: Dict[str, Any] = {}
        for embed_name, embed_info in embeddings.items():
            if isinstance(embed_info, str):
                normalised_embeddings[embed_name] = {
                    'path': _normalise_relative_path(embed_info)
                }
            else:
                normalised = dict(embed_info)
                if 'path' in normalised:
                    normalised['path'] = _normalise_relative_path(normalised['path'])
                if 'splits' in normalised and isinstance(normalised['splits'], dict):
                    normalised['splits'] = {
                        name: value for name, value in normalised['splits'].items()
                    }
                normalised_embeddings[embed_name] = normalised
        entry['embeddings'] = normalised_embeddings

        return entry


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))


def _normalise_relative_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path.lstrip('/')


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
