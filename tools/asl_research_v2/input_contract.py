"""Experimental isolated inference boundary; never connected to Android."""
import numpy as np
def validate(features: np.ndarray,mask: np.ndarray,schema: str) -> None:
    if schema!='3.0':raise ValueError('EXPERIMENT_SCHEMA_MISMATCH')
    if features.dtype!=np.float32 or features.ndim!=3 or features.shape[-1]!=258:raise ValueError('EXPERIMENT_FEATURE_SHAPE_OR_DTYPE')
    if not 1<=features.shape[1]<=96:raise ValueError('EXPERIMENT_ISOLATED_LENGTH_OUT_OF_BOUNDS')
    if mask.dtype!=np.bool_ or mask.shape!=features.shape[:2]:raise ValueError('EXPERIMENT_MASK')
    if not np.isfinite(features).all() or not mask.any(1).all():raise ValueError('EXPERIMENT_EMPTY_OR_NONFINITE')
