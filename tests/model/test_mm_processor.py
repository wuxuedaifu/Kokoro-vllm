import numpy as np
import pytest
import torch
from kokoro_vllm.model.mm_processor import normalize_ref_s


def test_from_ndarray():
    r = normalize_ref_s(np.ones(256, dtype=np.float32))
    assert isinstance(r, torch.Tensor)
    assert r.shape == (256,)
    assert r.dtype == torch.float32


def test_from_list():
    r = normalize_ref_s([0.0] * 256)
    assert r.shape == (256,)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        normalize_ref_s(np.ones(128, dtype=np.float32))
