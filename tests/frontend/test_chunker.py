from kokoro_vllm.frontend.chunker import chunk_text, Chunk

class FakeG2P:
    # 1 phoneme char per input char, deterministic
    def phonemize(self, text):
        return text.replace(" ", "").replace(".", "")

# vocab maps every lowercase letter to a distinct id
VOCAB = {chr(c): c for c in range(ord("a"), ord("z") + 1)}

def test_short_text_single_chunk():
    chunks = chunk_text("ab. cd.", FakeG2P(), VOCAB, max_tokens=510)
    assert len(chunks) == 2                      # two sentences
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].input_ids[0] == 0 and chunks[0].input_ids[-1] == 0

def test_respects_max_tokens():
    long = " ".join(["abcde"] * 100) + "."       # 500 phoneme chars, one sentence
    chunks = chunk_text(long, FakeG2P(), VOCAB, max_tokens=120)
    for c in chunks:
        assert len(c.input_ids) <= 120 + 2

def test_never_empty_chunks():
    chunks = chunk_text("a.  . b.", FakeG2P(), VOCAB, max_tokens=510)
    assert all(len(c.input_ids) > 2 for c in chunks)   # more than just padding
