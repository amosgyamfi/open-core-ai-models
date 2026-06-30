"""VibeVoice -> Core AI export package.

Converts the components of microsoft/VibeVoice-Realtime-0.5B into standalone
Core AI ``.aimodel`` assets that are orchestrated at runtime by the
``CoreAIVibeVoice`` Swift package.

The VibeVoice streaming TTS pipeline is decomposed into these exportable graphs:

* ``diffusion_head``     - per-frame DPM denoiser (noisy latent, t, condition) -> velocity
* ``acoustic_decoder``   - single acoustic latent + conv-cache state -> 3200 audio samples
* ``acoustic_connector`` - acoustic latent (64) -> LM embedding (896)
* ``eos_classifier``     - LM hidden state (896) -> EOS logit
* ``token_embed``        - token id -> embedding (896)
* ``base_lm``            - 4-layer Qwen2 stack (text encoder) + KV cache
* ``tts_lm``             - 20-layer Qwen2 stack (acoustic backbone) + KV cache

See ``README.md`` for the full architecture and the runtime loop.
"""

__all__ = ["common", "components"]
