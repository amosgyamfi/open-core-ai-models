# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Re-authored Pocket-TTS components that lower cleanly to Core AI `.aimodel` bundles.

Three on-engine graphs + host-side glue, mirroring the validated VoxCPM diffusion-TTS port:

  - ``Backbone``      q=1 stateful KV decode (prefill-via-decode): inputs_embeds -> hidden
  - ``FlowDecoder``   single-step LSD flow: (cond, noise) -> latent
  - ``MimiDecoder``   stateless full-sequence codec decode: latents -> waveform

The autoregressive control loop, text tokenization/embedding, latent projection, BOS handling,
EOS head and noise sampling run host-side (Python here / Swift on device).
"""

from .model import (  # noqa: F401
    Backbone,
    Components,
    FlowDecoder,
    MimiDecoder,
    load_components,
)
