"""GPU parity gate: vLLM engine path vs. reference kokoro.KModel.

Runs the *same* `(phonemes, ref_s, speed)` through:
  1. the vLLM V1 pooling engine (loads the converted `model.safetensors`), and
  2. the reference `kokoro.KModel` (loads the original `kokoro-v1_0.pth`),
and asserts the two waveforms are near-identical (correlation > 0.99).

Skips cleanly when the converted weights are absent (no GPU / not staged).
Requires a single visible CUDA GPU: set `CUDA_VISIBLE_DEVICES=0`.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

MODEL_DIR = os.getenv("KOKORO_MODEL_DIR", "./kokoro-model")
VOICES = os.getenv("KOKORO_VOICES_PATH", "./voices-v1.0.bin")
VOCAB = os.getenv("KOKORO_VOCAB_PATH", "./kokoro-model/config.json")
PTH = os.getenv("KOKORO_PTH_PATH", "./kokoro-model/kokoro-v1_0.pth")

SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")


@pytest.mark.skipif(
    not os.path.exists(SAFETENSORS),
    reason="no converted weights (run scripts/convert_weights.py)",
)
def test_vllm_matches_reference_kmodel():
    from vllm import LLM, PoolingParams
    from vllm.inputs.llm import TokensPrompt

    from kokoro_vllm.frontend.vocab import load_vocab, phonemes_to_input_ids
    from kokoro_vllm.frontend.voices import load_voicepacks, select_ref_s
    from kokoro_vllm.model.register import register

    # Register the Kokoro arch + transformers config in this (main) process so
    # vLLM's ModelConfig can parse config.json. The engine subprocess registers
    # itself via the `vllm.general_plugins` entry point.
    register()

    vocab = load_vocab(VOCAB)
    packs = load_voicepacks(VOICES)
    phonemes = "hˈɛlˌoʊ"  # fixed phoneme string
    ids = phonemes_to_input_ids(phonemes, vocab)
    ref_s = select_ref_s(packs, "af_sarah", num_tokens=len(ids) - 2)
    ref_s = np.asarray(ref_s, dtype=np.float32).reshape(-1)

    # --- vLLM engine path ---
    llm = LLM(
        model=MODEL_DIR,
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=1,
        # Kokoro's ALBERT text encoder has max_position_embeddings=512; the
        # dummy profiling run must not exceed it. Phoneme prompts are <= 512.
        max_model_len=512,
        trust_remote_code=True,
        # We supply raw phoneme token ids, so no tokenizer is needed.
        skip_tokenizer_init=True,
        # Match the reference KModel's float32 precision for parity.
        dtype="float32",
        # FlashInfer autotune runs a dummy _dummy_run that indexes the model
        # output as a tensor; Kokoro's forward returns a list of per-request
        # waveforms, so disable it (harmless: Kokoro uses no FlashInfer attn).
        enable_flashinfer_autotune=False,
    )
    prompt = TokensPrompt(
        prompt_token_ids=ids,
        multi_modal_data={"voice": ref_s},
    )
    out = llm.encode(
        prompt,
        pooling_params=PoolingParams(task="plugin"),
        pooling_task="plugin",
    )
    vllm_audio = np.asarray(out[0].outputs.data, dtype=np.float32).reshape(-1)

    # --- reference KModel path (ground-truth original checkpoint) ---
    import torch

    from kokoro import KModel

    # Run the reference on the SAME device (CUDA) as the vLLM engine so the
    # comparison isolates the port itself, not CPU-vs-GPU float32 accumulation
    # differences through the deep LSTM / conv / iSTFT decoder stack.
    ref_device = "cuda" if torch.cuda.is_available() else "cpu"
    km = KModel(
        repo_id="hexgrad/Kokoro-82M", config=VOCAB, model=PTH
    ).eval().to(ref_device)
    # Kokoro's iSTFTNet decoder injects unseeded noise (kokoro.istftnet SineGen:
    # torch.rand / torch.randn_like), so the waveform is non-deterministic
    # run-to-run. The vLLM model seeds the RNG to KOKORO_SYNTHESIS_SEED before
    # synthesis; seed the reference identically so the comparison measures the
    # port's numerics, not the decoder's stochastic noise floor.
    from kokoro_vllm.model.kokoro_vllm_model import KOKORO_SYNTHESIS_SEED

    torch.manual_seed(KOKORO_SYNTHESIS_SEED)
    if ref_device == "cuda":
        torch.cuda.manual_seed_all(KOKORO_SYNTHESIS_SEED)
    with torch.no_grad():
        ref_audio, _ = km.forward_with_tokens(
            torch.tensor([ids], dtype=torch.long, device=ref_device),
            torch.tensor(ref_s, dtype=torch.float32, device=ref_device).unsqueeze(0),
            1.0,
        )
    ref_audio = ref_audio.reshape(-1).detach().cpu().numpy().astype(np.float32)

    n = min(len(vllm_audio), len(ref_audio))
    assert n > 0, "empty audio output"
    corr = np.corrcoef(vllm_audio[:n], ref_audio[:n])[0, 1]
    print(f"\n[parity] vLLM-vs-reference waveform correlation = {corr:.6f} "
          f"(n={n} samples)")
    assert corr > 0.99, (
        f"waveform correlation too low: {corr} "
        f"(vllm_len={len(vllm_audio)}, ref_len={len(ref_audio)})"
    )
