import json
from transformers import PretrainedConfig


class KokoroConfig(PretrainedConfig):
    model_type = "kokoro"

    def __init__(self, hidden_dim=512, vocab_size=178, max_phoneme_len=510,
                 sample_rate=24000, vocab=None, kmodel_kwargs=None, **kwargs):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_phoneme_len = max_phoneme_len
        self.sample_rate = sample_rate
        self.vocab = vocab or {}
        self.kmodel_kwargs = kmodel_kwargs or {}
        kwargs.setdefault("architectures", ["KokoroForConditionalGeneration"])
        super().__init__(**kwargs)


def build_hf_config(kokoro_config_path: str) -> KokoroConfig:
    with open(kokoro_config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return KokoroConfig(
        hidden_dim=raw.get("hidden_dim", 512),
        vocab_size=len(raw.get("vocab", {})) or raw.get("n_token", 178),
        vocab=raw.get("vocab", {}),
        kmodel_kwargs=raw,
    )
