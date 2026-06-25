"""Oracle parity gate for the re-authored LFM2.5 decoder.

Runs the fp32 HuggingFace model as the reference (full prefill) and steps the
re-authored Core AI decoder one token at a time from a fresh state, comparing the
next-token top-1 at every position plus per-position logit cosine similarity.
"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from coreai_models.models.macos.lfm2 import build_decode_state, lfm2_from_hf

PROMPT = "The capital of France is Paris. The largest planet in the solar system is"


def load_tokenizer(hf_id: str) -> PreTrainedTokenizerFast:
    # The HF tokenizer_config uses a transformers-5.x class the pinned 4.57 env
    # doesn't know; load the fast tokenizer straight from tokenizer.json instead.
    tok_json = hf_hub_download(hf_id, "tokenizer.json")
    return PreTrainedTokenizerFast(tokenizer_file=tok_json)


def quantize_int8lin(model, cfg):
    """Apply the int8lin ship recipe (matches export_lfm2_230m.py)."""
    import torch.export

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.compression import quantize_pytorch_model
    from coreai_models.primitives.macos.cache import KVCache

    from export_lfm2_230m import linear_quant_config  # local driver

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=torch.float16)
    ref = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=4095)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=4096)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=4096)
    dyn = {
        "input_ids": None,
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
    }
    cfg_q = linear_quant_config("int8", block=32)
    return quantize_pytorch_model(model, tuple(ref.values()), dyn, cfg_q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-230M")
    ap.add_argument("--mode", default="fp16", choices=["fp16", "int8lin"])
    args = ap.parse_args()
    hf_id = args.hf_id

    tok = load_tokenizer(hf_id)
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(torch.int32)
    seq = input_ids.shape[1]
    print(f"prompt tokens: {seq}  mode: {args.mode}")

    print("loading fp32 HF reference ...")
    hf = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32).eval()
    with torch.no_grad():
        ref_logits = hf(input_ids.to(torch.long)).logits[0].float()  # [S, V]
    ref_top1 = ref_logits.argmax(-1)
    # Per-position fp32 top-2 margin: positions below the confidence threshold are
    # statistical ties (flip on healthy quant noise) and are excluded from the gate.
    top2 = ref_logits.topk(2, dim=-1).values
    ref_margin = (top2[:, 0] - top2[:, 1])
    del hf
    MARGIN = 0.1

    print("loading re-authored fp16 decoder ...")
    model = lfm2_from_hf(hf_id, target_dtype=torch.float16, stateful=True)
    cfg = model.config
    if args.mode == "int8lin":
        print("quantizing int8lin ...")
        model = quantize_int8lin(model, cfg)
        model.eval()

    max_seq = seq + 2
    state = build_decode_state(cfg, max_seq_len=max_seq, dtype=torch.float16)
    k_cache, v_cache, conv_state = state["k_cache"], state["v_cache"], state["conv_state"]

    match = 0
    scored = 0
    cos_sum = 0.0
    with torch.no_grad():
        for t in range(seq):
            ids_t = input_ids[:, t : t + 1]
            pos_t = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
            logits_t = model(ids_t, pos_t, k_cache, v_cache, conv_state)[0, -1].float()
            pred = int(logits_t.argmax(-1))
            ref = int(ref_top1[t])
            confident = float(ref_margin[t]) >= MARGIN
            cos = torch.nn.functional.cosine_similarity(
                logits_t, ref_logits[t], dim=0
            ).item()
            cos_sum += cos
            if confident:
                scored += 1
                match += pred == ref
                flag = "" if pred == ref else "  <-- MISMATCH"
            else:
                flag = "  (tie, excluded)"
            print(
                f"pos {t:2d}: mine={pred:6d} ref={ref:6d} "
                f"margin={float(ref_margin[t]):6.3f} cos={cos:.6f}{flag}"
            )

    print(
        f"\n[{args.mode}] confident top-1 parity: {match}/{scored}"
        f"   mean logit cosine: {cos_sum / seq:.6f}"
    )
    print("ORACLE GATE: PASS" if match == scored else "ORACLE GATE: FAIL")


if __name__ == "__main__":
    main()
