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
    #
    # The real kokoro-v1_0.pth is a dict of the five KModel submodule names
    # (bert, bert_encoder, predictor, text_encoder, decoder), each mapping to
    # that submodule's state_dict. Those nested keys carry a leading "module."
    # prefix left over from a DataParallel wrapper at training time
    # (e.g. "module.embeddings.word_embeddings.weight"). KModel.__init__ itself
    # strips this prefix (its except-branch does `{k[7:]: v ...}`) before
    # load_state_dict, so the real submodule parameter names have NO "module."
    # prefix. We must strip it here too, otherwise the converted key
    # "bert.module.embeddings..." never matches KModel's "bert.embeddings...".
    flat = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk.startswith("module."):
                    kk = kk[len("module."):]
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    flat = {k: v.contiguous() for k, v in flat.items() if torch.is_tensor(v)}
    save_file(flat, os.path.join(args.out, "model.safetensors"))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Preserve the full raw Kokoro config under `kmodel_kwargs` so that when
    # vLLM instantiates KokoroConfig directly from this config.json (via
    # transformers AutoConfig, NOT via build_hf_config), `config.kmodel_kwargs`
    # is still populated with everything KModel.__init__ needs (vocab, n_token,
    # plbert, hidden_dim, style_dim, n_layer, max_dur, dropout,
    # text_encoder_kernel_size, n_mels, istftnet). Without this, build_kmodel
    # raises KeyError('vocab') at model-init time in the engine.
    raw = dict(cfg)
    cfg["kmodel_kwargs"] = raw
    cfg["hidden_dim"] = raw.get("hidden_dim", 512)
    cfg["vocab_size"] = len(raw.get("vocab", {})) or raw.get("n_token", 178)
    cfg["architectures"] = ["KokoroForConditionalGeneration"]
    cfg["model_type"] = "kokoro"
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"Wrote {args.out}/model.safetensors ({len(flat)} tensors)")


if __name__ == "__main__":
    main()
