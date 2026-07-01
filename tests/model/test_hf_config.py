import json
from kokoro_vllm.model.hf_config import KokoroConfig, build_hf_config

def test_build(tmp_path):
    cfg = {"vocab": {"a": 1}, "hidden_dim": 512, "n_token": 178}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    c = build_hf_config(str(p))
    assert isinstance(c, KokoroConfig)
    assert c.architectures == ["KokoroForConditionalGeneration"]
    assert c.model_type == "kokoro"
    assert c.sample_rate == 24000
    assert c.max_phoneme_len == 510
    assert c.vocab == {"a": 1}
