import pytest
from kokoro_vllm.frontend import g2p as g2p_mod
from kokoro_vllm.frontend.g2p import G2P

def test_phonemize_delegates(monkeypatch):
    # Replace the backend factory so no model download happens
    class FakeBackend:
        def __call__(self, text):
            return ("hɛlo", None)   # misaki returns (phonemes, tokens)
    monkeypatch.setattr(g2p_mod, "_build_backend", lambda lang: FakeBackend())
    g = G2P(lang="en-us")
    assert g.phonemize("hello") == "hɛlo"

def test_unsupported_lang_raises():
    with pytest.raises(ValueError):
        G2P(lang="kl-fake")
