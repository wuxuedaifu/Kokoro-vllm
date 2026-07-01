import argparse
import json
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to kokoro-v1_0.pth")
    ap.add_argument("--config", required=True, help="Path to Kokoro config.json")
    ap.add_argument("--out", required=True, help="Output model dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state = torch.load(args.pth, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    # Flatten KModel's per-submodule dicts if present.
    flat = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    flat = {k: v.contiguous() for k, v in flat.items() if torch.is_tensor(v)}
    save_file(flat, os.path.join(args.out, "model.safetensors"))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["KokoroForConditionalGeneration"]
    cfg["model_type"] = "kokoro"
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"Wrote {args.out}/model.safetensors ({len(flat)} tensors)")


if __name__ == "__main__":
    main()
