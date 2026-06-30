#!/usr/bin/env python
"""Measure PyTorch VibeVoice streaming duration variance across RNG seeds.

Compares against the Swift Core AI pipeline's early-EOS behaviour: we want to
know whether the reference model itself produces wildly different durations per
seed for a short prompt, or whether it is stable (which would point to a bug in
the Swift port).

Run from the VibeVoice repo root (so `vibevoice` is importable):

    cd VibeVoice && uv run python ../VibeVoiceCoreAI/export/ref_seed_sweep.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "VibeVoice")
sys.path.insert(0, os.path.abspath(REPO))

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402

from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # noqa: E402
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (  # noqa: E402
    VibeVoiceStreamingProcessor,
)

MODEL = "microsoft/VibeVoice-Realtime-0.5B"
VOICE = os.path.abspath(os.path.join(REPO, "demo/voices/streaming_model/en-Carter_man.pt"))
TEXT = "Hello from on device Core AI."
SEEDS = [42, 1, 7, 123, 2024]
SR = 24000


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa", device_map=None
    )
    model.to(device).eval()
    model.set_ddpm_inference_steps(num_steps=5)

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefilled = torch.load(VOICE, map_location=device, weights_only=True)

    text = TEXT.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    inputs = processor.process_input_with_cached_prompt(
        text=text, cached_prompt=prefilled, padding=True,
        return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    print(f"tts_text_ids ({inputs['tts_text_ids'].shape[1]}): "
          f"{inputs['tts_text_ids'][0].tolist()}")

    # Instrument: log frame-0 diffusion condition (deterministic) + per-frame EOS.
    import types
    orig_sample = model.sample_speech_tokens
    orig_eos = model.tts_eos_classifier
    state = {"frame": 0}

    def patched_sample(self, condition, neg_condition, cfg_scale=3.0):
        if state["frame"] < 3:
            c = condition[0]
            print(f"  [f{state['frame']}] cond norm={c.norm().item():.4f} "
                  f"head={[round(x, 4) for x in c[:6].tolist()]}")
            nc = neg_condition[0]
            print(f"        neg  norm={nc.norm().item():.4f} "
                  f"head={[round(x, 4) for x in nc[:6].tolist()]}")
        return orig_sample(condition, neg_condition, cfg_scale=cfg_scale)

    def base_hook(module, inp, out):
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if h.shape[1] == 5 and not state.get("base_done"):
            r = h[0, -1, :]
            print(f"  BASE lm out[-1] norm={r.norm().item():.4f} "
                  f"head={[round(x, 4) for x in r[:6].tolist()]}")
            state["base_done"] = True

    bhandle = model.model.language_model.register_forward_hook(base_hook)

    def eos_hook(module, inp, out):
        p = torch.sigmoid(out).flatten()[0].item()
        if state["frame"] < 12:
            print(f"  [f{state['frame']}] eos={p:.4f}")
        state["frame"] += 1

    model.sample_speech_tokens = types.MethodType(patched_sample, model)
    handle = model.tts_eos_classifier.register_forward_hook(eos_hook)

    for seed in SEEDS[:1]:
        state["frame"] = 0
        torch.manual_seed(seed)
        out = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False}, verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled),
        )
        wav = out.speech_outputs[0]
        n = wav.shape[-1] if wav is not None else 0
        print(f"INSTRUMENTED seed={seed} duration={n / SR:.2f}s")

    model.sample_speech_tokens = orig_sample
    handle.remove()
    bhandle.remove()

    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        out = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False}, verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled),
        )
        wav = out.speech_outputs[0]
        n = wav.shape[-1] if wav is not None else 0
        print(f"seed={seed:<5} duration={n / SR:5.2f}s  samples={n}")


if __name__ == "__main__":
    main()
