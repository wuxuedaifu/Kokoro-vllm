import logging
import re
from dataclasses import dataclass

from kokoro_vllm.frontend.vocab import phonemes_to_input_ids

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")


@dataclass
class Chunk:
    phonemes: str
    input_ids: list[int]


def _token_len(phonemes: str, vocab: dict) -> int:
    # count of mappable phonemes (excludes the two padding zeros)
    return sum(1 for p in phonemes if p in vocab)


def _emit(phonemes: str, vocab: dict) -> Chunk | None:
    ids = phonemes_to_input_ids(phonemes, vocab)
    if len(ids) <= 2:  # only padding -> nothing real
        return None
    return Chunk(phonemes=phonemes, input_ids=ids)


def chunk_text(text, g2p, vocab, max_tokens=510):
    chunks: list[Chunk] = []
    for sentence in _SENTENCE_RE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        phon = g2p.phonemize(sentence)
        if _token_len(phon, vocab) <= max_tokens:
            c = _emit(phon, vocab)
            if c:
                chunks.append(c)
            continue
        # sentence too long: split on words
        buf = ""
        for word in sentence.split():
            wphon = g2p.phonemize(word)
            if _token_len(wphon, vocab) > max_tokens:
                # Flush whatever we've accumulated so far first.
                c = _emit(g2p.phonemize(buf), vocab)
                if c:
                    chunks.append(c)
                buf = ""
                logger.warning("Word exceeds max_tokens; hard-truncating: %r", word)
                truncated = "".join(p for p in wphon if p in vocab)[:max_tokens]
                c = _emit(truncated, vocab)
                if c:
                    chunks.append(c)
                continue
            candidate = (buf + " " + word).strip()
            cphon = g2p.phonemize(candidate)
            if _token_len(cphon, vocab) > max_tokens:
                c = _emit(g2p.phonemize(buf), vocab)
                if c:
                    chunks.append(c)
                buf = word
            else:
                buf = candidate
        c = _emit(g2p.phonemize(buf), vocab)
        if c:
            chunks.append(c)
    return chunks
