LANG_TO_MISAKI = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "fr-fr": "fr-fr",
    "it": "it",
    "ja": "ja",
    "cmn": "cmn",
}


def _build_backend(lang: str):
    # Imported lazily so unit tests can monkeypatch this factory.
    if lang in ("en-us", "en-gb"):
        from misaki import en
        british = lang == "en-gb"
        return en.G2P(british=british)
    from misaki import espeak
    return espeak.EspeakG2P(language=LANG_TO_MISAKI[lang])


class G2P:
    def __init__(self, lang: str = "en-us"):
        if lang not in LANG_TO_MISAKI:
            raise ValueError(
                f"Unsupported language: {lang}. "
                f"Supported: {', '.join(sorted(LANG_TO_MISAKI))}"
            )
        self.lang = lang
        self._backend = _build_backend(lang)

    def phonemize(self, text: str) -> str:
        phonemes, _ = self._backend(text)
        return phonemes
