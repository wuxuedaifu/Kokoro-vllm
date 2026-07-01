import numpy as np


def load_voicepacks(voices_path: str) -> dict[str, np.ndarray]:
    # voices-v1.0.bin is a np.savez-style archive of {name: [510,(1,)256]}
    data = np.load(voices_path, allow_pickle=True)
    packs = {}
    for name in data.files:
        arr = np.asarray(data[name], dtype=np.float32)
        if arr.ndim == 3:            # [510,1,256] -> [510,256]
            arr = arr.squeeze(1)
        packs[name] = arr
    return packs


def list_voices(packs: dict[str, np.ndarray]) -> list[str]:
    return list(packs.keys())


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty voice spec")
    if len(parts) > 2:
        raise ValueError("Voice blending supports at most two voices")
    out = []
    for part in parts:
        if ":" in part:
            name, w = part.split(":")
            out.append((name.strip(), float(w.strip())))
        else:
            out.append((part, 50.0 if len(parts) == 2 else 100.0))
    total = sum(w for _, w in out)
    if len(parts) == 1:
        return [(out[0][0], 100.0)]
    if total <= 0:
        raise ValueError("blend weights must sum to a positive value")
    return [(n, w * 100.0 / total) for n, w in out]


def select_ref_s(packs, spec, num_tokens):
    idx = min(max(num_tokens, 0), 509)
    blend = parse_voice_spec(spec)
    ref = np.zeros(256, dtype=np.float32)
    for name, weight in blend:
        if name not in packs:
            raise ValueError(
                f"Unsupported voice: {name}. "
                f"Available: {', '.join(sorted(packs))}"
            )
        ref += (weight / 100.0) * packs[name][idx]
    return ref.astype(np.float32)
