import json

MAX_TOKENS = 510


def load_vocab(vocab_path: str) -> dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["vocab"]


def phonemes_to_input_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab[p] for p in phonemes if p in vocab]
    return [0, *ids, 0]
