import os
from dataclasses import dataclass


@dataclass
class ServerSettings:
    model_dir: str = "./kokoro-model"
    voices_path: str = "./voices-v1.0.bin"
    vocab_path: str = "./config.json"
    device: str = "cuda"
    max_num_seqs: int = 64
    sample_rate: int = 24000
    max_phoneme_tokens: int = 510

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            model_dir=os.getenv("KOKORO_MODEL_DIR", cls.model_dir),
            voices_path=os.getenv("KOKORO_VOICES_PATH", cls.voices_path),
            vocab_path=os.getenv("KOKORO_VOCAB_PATH", cls.vocab_path),
            device=os.getenv("KOKORO_DEVICE", cls.device),
            max_num_seqs=int(os.getenv("KOKORO_MAX_NUM_SEQS", cls.max_num_seqs)),
            sample_rate=int(os.getenv("KOKORO_SAMPLE_RATE", cls.sample_rate)),
            max_phoneme_tokens=int(
                os.getenv("KOKORO_MAX_PHONEME_TOKENS", cls.max_phoneme_tokens)
            ),
        )
