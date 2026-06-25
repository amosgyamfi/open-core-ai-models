"""End-to-end check: run the exported .aimodel through the Core AI runtime and
compare a teacher-forced decode sweep against the fp32 HF oracle.

This validates the *exported graph* (torch.export -> Core AI -> optimize -> save),
not just the torch model. States (keyCache, valueCache, convState) are created as
NDArrays and reused/mutated across decode steps.
"""

import argparse
import asyncio
import inspect

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

import coreai.runtime as rt
from coreai_models.models.macos.lfm2 import lfm2_hf_config

PROMPT = "The capital of France is Paris. The largest planet in the solar system is"


def load_tokenizer(hf_id):
    return PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(hf_id, "tokenizer.json"))


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-230M")
    ap.add_argument("--unit", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    cfg = lfm2_hf_config(args.hf_id)
    tok = load_tokenizer(args.hf_id)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(torch.int32)
    seq = ids.shape[1]

    hf = AutoModelForCausalLM.from_pretrained(args.hf_id, dtype=torch.float32).eval()
    with torch.no_grad():
        ref_logits = hf(ids.to(torch.long)).logits[0].float()
    ref_top1 = ref_logits.argmax(-1)
    top2 = ref_logits.topk(2, -1).values
    margin = top2[:, 0] - top2[:, 1]
    del hf

    opts = (
        rt.SpecializationOptions.cpu_only()
        if args.unit == "cpu"
        else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    )
    aimodel = await rt.AIModel.load(args.bundle, opts)
    func = aimodel.load_function("main")
    print("call signature:", inspect.signature(func.__call__))
    print("desc:", {a: getattr(func.desc, a) for a in dir(func.desc) if not a.startswith("_")})

    max_seq = 2048
    n_attn = cfg.num_full_layers
    n_conv = cfg.num_conv_layers
    k = np.zeros((n_attn, 1, cfg.num_key_value_heads, max_seq, cfg.head_dim), np.float16)
    v = np.zeros_like(k)
    conv = np.zeros((n_conv, 1, cfg.hidden_size, cfg.conv_L_cache - 1), np.float16)
    # Create state NDArrays once and reuse them; the runtime mutates them in place
    # across decode steps (growing KV + rolling conv history).
    state = {
        "keyCache": rt.NDArray(data=k),
        "valueCache": rt.NDArray(data=v),
        "convState": rt.NDArray(data=conv),
    }

    match = scored = 0
    for t in range(seq):
        in_ids = ids[:, t : t + 1].numpy().astype(np.int32)
        pos = np.arange(t + 1, dtype=np.int32)[None]
        inputs = {
            "input_ids": rt.NDArray(data=in_ids),
            "position_ids": rt.NDArray(data=pos),
        }
        out = await func(inputs=inputs, state=state)
        logits = np.asarray(next(iter(out.values())).numpy()).reshape(-1)
        pred = int(logits.argmax())
        ref = int(ref_top1[t])
        if float(margin[t]) >= 0.1:
            scored += 1
            match += pred == ref
            flag = "" if pred == ref else "  <-- MISMATCH"
        else:
            flag = "  (tie, excluded)"
        print(f"pos {t:2d}: aimodel={pred:6d} ref={ref:6d} margin={float(margin[t]):6.3f}{flag}")

    print(f"\n[.aimodel/{args.unit}] confident top-1 parity: {match}/{scored}")
    print("ENGINE GATE: PASS" if match == scored else "ENGINE GATE: FAIL")


if __name__ == "__main__":
    asyncio.run(amain())
