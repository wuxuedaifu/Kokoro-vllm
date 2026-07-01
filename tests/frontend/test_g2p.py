import pytest
from kokoro_vllm.frontend import g2p as g2p_mod
from kokoro_vllm.frontend.g2p import G2P, cached_g2p_factory

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

def test_cached_factory_builds_backend_once(monkeypatch):
    builds = {"n": 0}
    class FakeBackend:
        def __call__(self, text):
            return ("x", None)
    def fake_build(lang):
        builds["n"] += 1
        return FakeBackend()
    monkeypatch.setattr(g2p_mod, "_build_backend", fake_build)

    factory = cached_g2p_factory()
    a1 = factory("en-us")
    a2 = factory("en-us")
    b1 = factory("fr-fr")
    assert a1 is a2                 # same lang -> same cached instance
    assert b1 is not a1             # different lang -> different instance
    assert builds["n"] == 2         # backend built once per distinct lang, not per call
