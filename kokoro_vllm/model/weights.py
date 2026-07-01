_PREFIX = "kmodel."


def remap_param_name(kmodel_name: str) -> str:
    """Map a raw kokoro.KModel state-dict key to its name under our vLLM
    module tree, where the real KModel is nested as `self.kmodel`.

    Pure string function: deterministic and unit-testable without weights.
    """
    if kmodel_name.startswith(_PREFIX):
        return kmodel_name
    return _PREFIX + kmodel_name
