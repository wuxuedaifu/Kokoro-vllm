import numpy as np
import pytest
from kokoro_vllm.frontend.voices import parse_voice_spec, select_ref_s, list_voices

def _packs():
    a = np.ones((510, 256), dtype=np.float32)
    b = np.full((510, 256), 3.0, dtype=np.float32)
    return {"af_sarah": a, "am_adam": b}

def test_parse_single():
    assert parse_voice_spec("af_sarah") == [("af_sarah", 100.0)]

def test_parse_blend_normalizes():
    out = parse_voice_spec("af_sarah:60,am_adam:40")
    assert out == [("af_sarah", 60.0), ("am_adam", 40.0)]

def test_parse_rejects_three():
    with pytest.raises(ValueError):
        parse_voice_spec("a,b,c")

def test_select_single_ref_s_shape_and_index():
    r = select_ref_s(_packs(), "af_sarah", num_tokens=5)
    assert r.shape == (256,)
    assert r.dtype == np.float32
    assert np.allclose(r, 1.0)

def test_select_blend_math():
    r = select_ref_s(_packs(), "af_sarah:50,am_adam:50", num_tokens=5)
    assert np.allclose(r, 0.5 * 1.0 + 0.5 * 3.0)  # == 2.0

def test_index_clamped():
    r = select_ref_s(_packs(), "af_sarah", num_tokens=9999)
    assert np.allclose(r, 1.0)  # clamped to 509, still valid

def test_list_voices():
    assert sorted(list_voices(_packs())) == ["af_sarah", "am_adam"]

def test_parse_rejects_zero_sum_blend():
    with pytest.raises(ValueError):
        parse_voice_spec("af_sarah:0,am_adam:0")
