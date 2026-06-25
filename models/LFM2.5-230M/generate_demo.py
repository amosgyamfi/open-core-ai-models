"""Greedy generation demo driving the exported LFM2.5-230M .aimodel through the
Core AI runtime (decode-only S=1 bundle: prompt is fed one token at a time, then
generation continues)."""

import argparse
import asyncio

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

import coreai.runtime as rt
from coreai_models.models.macos.lfm2 import lfm2_hf_config

CHAT = (
    "<|startoftext|><|im_start|>system\n"
    "You are a helpful assistant trained by Liquid AI.<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-230M")
    ap.add_argument("--prompt", default="In one sentence, what is C. elegans?")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    cfg = lfm2_hf_config(args.hf_id)
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(args.hf_id, "tokenizer.json"))
    ids = tok(CHAT.format(user=args.prompt), return_tensors="pt").input_ids[0].tolist()

    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    model = await rt.AIModel.load(args.bundle, opts)
    func = model.load_function("main")

    max_seq = 2048
    state = {
        "keyCache": rt.NDArray(
            data=np.zeros(
                (cfg.num_full_layers, 1, cfg.num_key_value_heads, max_seq, cfg.head_dim), np.float16
            )
        ),
        "valueCache": rt.NDArray(
            data=np.zeros(
                (cfg.num_full_layers, 1, cfg.num_key_value_heads, max_seq, cfg.head_dim), np.float16
            )
        ),
        "convState": rt.NDArray(
            data=np.zeros((cfg.num_conv_layers, 1, cfg.hidden_size, cfg.conv_L_cache - 1), np.float16)
        ),
    }

    async def step(token, pos):
        inputs = {
            "input_ids": rt.NDArray(data=np.array([[token]], np.int32)),
            "position_ids": rt.NDArray(data=np.arange(pos + 1, dtype=np.int32)[None]),
        }
        out = await func(inputs=inputs, state=state)
        return np.asarray(next(iter(out.values())).numpy()).reshape(-1)

    # Prefill the prompt (S=1 steps), keep the last logits.
    logits = None
    for pos, t in enumerate(ids):
        logits = await step(t, pos)

    out_ids = []
    pos = len(ids)
    for _ in range(args.max_new):
        nxt = int(logits.argmax())
        if nxt == cfg_eos(cfg):
            break
        out_ids.append(nxt)
        logits = await step(nxt, pos)
        pos += 1

    print("PROMPT:", args.prompt)
    print("OUTPUT:", tok.decode(out_ids))


def cfg_eos(cfg):
    return 7  # LFM2.5 <|im_end|>


if __name__ == "__main__":
    asyncio.run(amain())
