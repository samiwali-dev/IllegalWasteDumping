"""Shared boundary: official test videos cannot enter training or tuning."""
from functools import lru_cache
from data import load_manifest, split_records


@lru_cache(maxsize=1)
def development_records():
    fit, validation, held_out = split_records(load_manifest(), seed=42)
    groups = [set(r['video_id'] for r in rows) for rows in (fit, validation, held_out)]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Video leakage between fit, validation and official test')
    if any(r['split'] != 'train' for r in fit + validation):
        raise ValueError('Official test record in a development split')
    return fit, validation


def require_development_video(video_id, role):
    if role not in ('fit', 'validation'):
        raise ValueError('Only fit and validation are permitted for development')
    fit, validation = development_records()
    allowed = {r['video_id'] for r in (fit if role == 'fit' else validation)}
    if video_id not in allowed:
        raise ValueError(f'{video_id} is not in the permitted {role} split')
