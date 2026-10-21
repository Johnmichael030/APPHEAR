"""Isolated recovery policy prototype; deliberately not wired into Android."""
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class MaterializationDecision:
    status: str
    missing_indices: tuple[int, ...]
    emit_sign: bool

def retained_event_decision(required_indices: Iterable[int], retained_indices: Iterable[int]) -> MaterializationDecision:
    """Discard an expired committed event, never truncate or synthesize its start.

    Caller must log EXPIRED_EVENT, skip recognition/SentenceBuilder and continue
    processing genuine later observations. No confidence threshold is invented.
    """
    required=tuple(required_indices); retained=set(retained_indices)
    if len(required)!=4 or any(i<0 for i in required): raise ValueError('EVENT_REQUIRED_INDICES')
    start,end,post,commit=required
    if not start<=end<post<=commit: raise ValueError('EVENT_INDEX_ORDER')
    missing=tuple(i for i in required if i not in retained)
    return MaterializationDecision('EXPIRED_EVENT' if missing else 'MATERIALIZE',missing,False)
