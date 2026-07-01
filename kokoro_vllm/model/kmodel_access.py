import tempfile

import torch

_DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"


def build_kmodel(config, device: str) -> torch.nn.Module:
    """Instantiate the real kokoro.KModel submodule graph on `device`.

    kokoro.KModel.__init__ *unconditionally* loads a state dict at
    construction time: if the `model` argument is falsy, it downloads the
    matching .pth from the Hugging Face Hub via `hf_hub_download`. There is
    no public constructor mode that builds the module graph without either a
    local weights file or a network download.

    To avoid any network access here (Task 8 must not fetch multi-GB
    weights; real weight loading happens later via Task 11's vLLM
    `load_weights`, using `remap_param_name` against a locally converted
    safetensors file, or Task 12's GPU parity harness), we hand KModel a
    local, empty state dict. Its per-submodule load loop
    (`for key, state_dict in torch.load(model, ...).items()`) then simply
    does nothing, so KModel still builds its real submodules -- bert,
    bert_encoder, predictor, text_encoder, decoder -- with
    randomly-initialized weights, and performs no I/O beyond reading that
    local empty file.

    `config.kmodel_kwargs` must hold the raw Kokoro config.json dict (as
    produced by `kokoro_vllm.model.hf_config.build_hf_config`), containing
    the keys KModel.__init__ expects: vocab, n_token, plbert, hidden_dim,
    style_dim, n_layer, max_dur, dropout, text_encoder_kernel_size, n_mels,
    istftnet.
    """
    from kokoro import KModel

    cfg = config.kmodel_kwargs
    repo_id = cfg.get("repo_id", _DEFAULT_REPO_ID)

    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save({}, tmp.name)
        model = KModel(
            repo_id=repo_id,
            config=cfg,
            model=tmp.name,
            disable_complex=False,
        )

    return model.to(device).eval()
