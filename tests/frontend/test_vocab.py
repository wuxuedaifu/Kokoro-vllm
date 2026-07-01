import json
from kokoro_vllm.frontend.vocab import phonemes_to_input_ids, load_vocab, MAX_TOKENS

VOCAB = {"h": 5, "ɛ": 6, "l": 7, "o": 8}

def test_padding_and_mapping():
    ids = phonemes_to_input_ids("hɛllo", VOCAB)
    assert ids == [0, 5, 6, 7, 7, 8, 0]

def test_drops_unknown_phonemes():
    ids = phonemes_to_input_ids("hZo", VOCAB)  # Z not in vocab
    assert ids == [0, 5, 8, 0]

def test_max_tokens_constant():
    assert MAX_TOKENS == 510

def test_load_vocab(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"vocab": VOCAB}))
    assert load_vocab(str(p)) == VOCAB
