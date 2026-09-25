"""Read PWM mesh databases, including existing locally prepared databases."""
from pathlib import Path
import json

ASSETS = Path(__file__).resolve().parent / 'assets'
CATEGORIES = ('Table', 'StorageFurniture', 'WashingMachine', 'Microwave',
              'Dishwasher', 'Refrigerator', 'Oven')


def resolve_hashbook(database_root, explicit=None):
    path = (Path(explicit).expanduser() if explicit is not None else
            Path(database_root) / 'pwm_hash_filtered.json')
    if explicit is None and not path.exists():
        path = ASSETS / 'pwm_hash_filtered.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing retrieval index: {path}')
    return path.resolve()


def load_hashbook(path):
    with Path(path).open(encoding='utf-8') as stream:
        book = json.load(stream)
    if not isinstance(book, dict):
        raise ValueError('Expected category -> hash -> object IDs in the retrieval index.')
    for category, groups in book.items():
        if not isinstance(category, str) or Path(category).name != category or category in ('.', '..'):
            raise ValueError('Invalid category in retrieval index.')
        if not isinstance(groups, dict):
            raise ValueError(f'Invalid hash groups for {category}.')
        for code, ids in groups.items():
            if not isinstance(code, str) or not isinstance(ids, list):
                raise ValueError(f'Invalid hash group for {category}.')
            for oid in ids:
                if not isinstance(oid, str) or Path(oid).name != oid or oid in ('', '.', '..'):
                    raise ValueError(f'Invalid object ID in {category}.')
    return book


def candidate_ids(book, category, graph_hash=None):
    groups = book.get(category, {})
    if graph_hash is not None:
        return sorted(set(groups.get(graph_hash, [])))
    return sorted({oid for ids in groups.values() for oid in ids})


def object_json_path(directory, filename="object_pwm.json"):
    path = Path(directory) / filename
    if filename == "object_pwm.json" and not path.is_file():
        legacy = Path(directory) / "object_uwm.json"
        if legacy.is_file():
            return legacy
    return path
