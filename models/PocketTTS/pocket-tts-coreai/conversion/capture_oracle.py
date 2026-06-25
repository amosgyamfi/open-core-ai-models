# Community port — NOT an Apple model.
"""Capture a deterministic oracle from the reference pocket-tts model for numeric gating.

Instruments the FlowLM backbone + flow_net + mimi decode of a real generation run and dumps
every intermediate tensor we need to gate the three Core AI bundles independently:

  - backbone:  per-step transformer input sequence (post text/latent embedding + cat) and the
               normalized hidden output (`out_norm(transformer_out)`), plus the write offset.
  - flow:      per-step (cond hidden, noise z, decoded latent) for the unrolled LSD loop.
  - mimi:      the full latent sequence fed to the codec and the decoded waveform.

Output: artifacts/oracle.npz  (+ _ref_alba.wav already written separately)

  python capture_oracle.py [--voice alba] [--text "..."] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pocket_tts import TTSModel
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models import flow_lm as flow_lm_mod

ART = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--text", default="Hello world, this is a test of pocket text to speech.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed)
    m = TTSModel.load_model()
    m.eval()

    rec: dict[str, list] = {
        "bb_in": [], "bb_hidden": [], "bb_offset": [],
        "flow_cond": [], "flow_noise": [], "flow_latent": [],
    }

    # ---- instrument the backbone transformer: record (input sequence, normalized output) ----
    # Wrap the transformer forward to grab the stateful call's input and out_norm'd output.
    fl = m.flow_lm
    orig_tf = fl.transformer.forward

    def tf_spy(x, model_state):
        offset = m._flow_lm_current_end(model_state) if model_state is not None else 0
        out = orig_tf(x, model_state)
        normed = fl.out_norm(out).to(torch.float32)
        rec["bb_in"].append(x.detach().to(torch.float32).cpu().numpy())
        rec["bb_offset"].append(int(offset))
        rec["bb_hidden"].append(normed.detach().cpu().numpy())
        return out

    fl.transformer.forward = tf_spy  # type: ignore[method-assign]

    # ---- instrument the flow net: record (cond, noise, latent) per decode call ----
    orig_lsd = flow_lm_mod.lsd_decode

    def lsd_spy(v_t, x_0, num_steps=1):
        # v_t == partial(self.flow_net, transformer_out); recover cond via closure.
        # NOTE: lsd_decode mutates x_0 in place (current = x_0; current += ...), so snapshot
        # the noise BEFORE calling it, otherwise the recorded "noise" becomes the latent.
        cond = v_t.args[0] if hasattr(v_t, "args") else None
        noise0 = x_0.detach().to(torch.float32).cpu().numpy().copy()
        latent = orig_lsd(v_t, x_0, num_steps)
        if cond is not None:
            rec["flow_cond"].append(cond.detach().to(torch.float32).cpu().numpy())
            rec["flow_noise"].append(noise0)
            rec["flow_latent"].append(latent.detach().to(torch.float32).cpu().numpy())
        return latent

    flow_lm_mod.lsd_decode = lsd_spy  # type: ignore[assignment]

    # ---- instrument mimi decode: record latent sequence + waveform ----
    orig_decode = m.mimi.decode_from_latent
    mimi_calls: list[tuple] = []

    def decode_spy(latent, mimi_state):
        out = orig_decode(latent, mimi_state)
        mimi_calls.append((latent.detach().to(torch.float32).cpu().numpy(),
                           out.detach().to(torch.float32).cpu().numpy()))
        return out

    m.mimi.decode_from_latent = decode_spy  # type: ignore[method-assign]

    # ---- run a real generation ----
    vs = m.get_state_for_audio_prompt(a.voice)
    torch.manual_seed(a.seed)
    audio = m.generate_audio(vs, a.text, copy_state=True)

    # also dump the prepared text tokens + the audio-prompt conditioning length
    prepared = fl.conditioner.prepare(a.text)

    out: dict[str, np.ndarray] = {}
    for i, x in enumerate(rec["bb_in"]):
        out[f"bb_in__{i}"] = x
        out[f"bb_hidden__{i}"] = rec["bb_hidden"][i]
    out["bb_offset"] = np.array(rec["bb_offset"], np.int64)
    out["n_bb"] = np.array([len(rec["bb_in"])], np.int64)
    for i, (c, n, l) in enumerate(zip(rec["flow_cond"], rec["flow_noise"], rec["flow_latent"])):
        out[f"flow_cond__{i}"] = c
        out[f"flow_noise__{i}"] = n
        out[f"flow_latent__{i}"] = l
    out["n_flow"] = np.array([len(rec["flow_cond"])], np.int64)
    for i, (lat, wav) in enumerate(mimi_calls):
        out[f"mimi_lat__{i}"] = lat
        out[f"mimi_wav__{i}"] = wav
    out["n_mimi"] = np.array([len(mimi_calls)], np.int64)
    out["audio"] = audio.detach().to(torch.float32).cpu().numpy()
    out["text_tokens"] = prepared.tokens.detach().cpu().numpy()
    np.savez(ART / "oracle.npz", **out)
    print(f"saved {ART/'oracle.npz'}: bb={len(rec['bb_in'])} flow={len(rec['flow_cond'])} "
          f"mimi={len(mimi_calls)} audio={audio.shape}")


if __name__ == "__main__":
    main()
