import os
from kokoro_vllm.config import ServerSettings

def test_defaults():
    s = ServerSettings()
    assert s.sample_rate == 24000
    assert s.max_phoneme_tokens == 510
    assert s.device == "cuda"
    assert s.vocab_path == "./kokoro-model/config.json"

def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("KOKORO_DEVICE", "cpu")
    monkeypatch.setenv("KOKORO_MAX_NUM_SEQS", "8")
    s = ServerSettings.from_env()
    assert s.device == "cpu"
    assert s.max_num_seqs == 8
