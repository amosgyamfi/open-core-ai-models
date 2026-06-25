---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
pipeline_tag: text-generation
tags:
- liquid
- lfm2.5
- edge
base_model: LiquidAI/LFM2.5-230M-Base
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-230M

LFM2.5 is a family of hybrid models designed for **on-device deployment**. It builds on the LFM2 architecture with extended pre-training and reinforcement learning.

- **Our most compact model yet**: 230M parameters that punch above their weight, bringing real capability to the tightest memory and compute budgets.
- **Fast edge inference**: Best throughput from low-cost CPUs to production GPUs, running at 213 tok/s decode speed on Galaxy S25 Ultra and 42 tok/s on a Raspberry Pi 5.
- **Built for agentic tasks**: Distilled from LFM2.5-350M and refined with multi-stage reinforcement learning, making it well-suited for tool use and data extraction.

Find more information about LFM2.5-230M in our [blog post](https://www.liquid.ai/blog/lfm2-5-230m).

![lfm2_5_230m_benchmarks](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/4UpNxlgfKjfgT5ByIVph0.png)

## 🗒️ Model Details

| Model | Parameters | Description |
|-------|------------|-------------|
| [LFM2.5-230M-Base](https://huggingface.co/LiquidAI/LFM2.5-230M-Base) | 230M | Pre-trained base model for fine-tuning |
| [**LFM2.5-230M**](https://huggingface.co/LiquidAI/LFM2.5-230M) | 230M | General-purpose instruction-tuned model |

LFM2.5-230M is a general-purpose text-only model with the following features:

- **Number of parameters**: 230M
- **Number of layers**: 14 (8 double-gated LIV convolution blocks + 6 GQA blocks)
- **Training budget**: 19T tokens
- **Context length**: 32,768 tokens
- **Vocabulary size**: 65,536
- **Knowledge cutoff**: Mid-2024
- **Languages**: English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish
- **Generation parameters**:
  - `temperature: 0.1`
  - `top_k: 50`
  - `repetition_penalty: 1.05`

| Model | Description |
|-------|-------------|
| [**LFM2.5-230M**](https://huggingface.co/LiquidAI/LFM2.5-230M) | Original model checkpoint in native format. Best for fine-tuning or inference with Transformers, vLLM, and SGLang. |
| [LFM2.5-230M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF) | Quantized format for llama.cpp and compatible tools. Optimized for edge inference and local deployment. |
| [LFM2.5-230M-ONNX](https://huggingface.co/LiquidAI/LFM2.5-230M-ONNX) | ONNX Runtime format for cross-platform deployment. |
| [LFM2.5-230M-MLX](https://huggingface.co/LiquidAI/LFM2.5-230M-MLX-8bit) | MLX format for Apple Silicon. Optimized for fast inference on Mac devices. |

We recommend using it for data extraction and lightweight on-device agentic pipelines. It is not recommended for reasoning-heavy workloads such as advanced math, code generation, or creative writing.

### Chat Template

LFM2.5 uses a ChatML-like format. See the [Chat Template documentation](https://docs.liquid.ai/lfm/key-concepts/chat-template) for details. Example:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
```

You can use [`tokenizer.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate) to format your messages automatically.

### Tool Use

LFM2.5 supports function calling in four steps:

1. **Function definition**: Provide the list of tools as a JSON object in the system prompt, or use [`tokenizer.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools) with `tools=...`.
2. **Function call**: By default, LFM2.5 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer. You can override this behavior by asking the model to output JSON function calls in the system prompt.
3. **Function execution**: Execute the call and return the result with the `tool` role.
4. **Final answer**: LFM2.5 interprets the tool output and returns a plain-text answer addressing the original prompt.

See the [Tool Use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for the full guide. Example:

```
<|startoftext|><|im_start|>system
List of tools: [{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

## 🏃 Inference

LFM2.5 is supported by many inference frameworks. See the [Inference documentation](https://docs.liquid.ai/lfm/inference/transformers) for the full list.

| Name | Description | Docs | Notebook |
|------|-------------|------|:--------:|
| [Transformers](https://github.com/huggingface/transformers) | Simple inference with direct access to model internals. | <a href="https://docs.liquid.ai/lfm/inference/transformers">Link</a> | <a href="https://colab.research.google.com/drive/1_q3jQ6LtyiuPzFZv7Vw8xSfPU5FwkKZY?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput production deployments with GPU. | <a href="https://docs.liquid.ai/lfm/inference/vllm">Link</a> | <a href="https://colab.research.google.com/drive/1VfyscuHP8A3we_YpnzuabYJzr5ju0Mit?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Cross-platform inference with CPU offloading. | <a href="https://docs.liquid.ai/lfm/inference/llama-cpp">Link</a> | <a href="https://colab.research.google.com/drive/1ohLl3w47OQZA4ELo46i5E4Z6oGWBAyo8?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [MLX](https://github.com/ml-explore/mlx) | Apple's machine learning framework optimized for Apple Silicon. | <a href="https://docs.liquid.ai/lfm/inference/mlx">Link</a> | — |
| [LM Studio](https://lmstudio.ai/) | Desktop application for running LLMs locally. | <a href="https://docs.liquid.ai/lfm/inference/lm-studio">Link</a> | — |
| [SGLang](https://github.com/vllm-project/vllm) | High-throughput production deployments with GPU. | <a href="https://docs.liquid.ai/deployment/gpu-inference/sglang">Link</a> | - </a> |


Quick start with Transformers (compatible with `transformers>=5.0.0`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "LiquidAI/LFM2.5-230M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "What is C. elegans?"

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
)["input_ids"].to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    max_new_tokens=512,
    streamer=streamer,
)
```

## 🔧 Fine-Tuning

We recommend fine-tuning LFM2.5 for your specific use case to achieve the best results.

| Name | Description | Docs | Notebook |
|------|-------------|------|----------|
| CPT ([Unsloth](https://github.com/unslothai/unsloth)) | Continued Pre-Training using Unsloth for text completion. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/10fm7eNMezs-DSn36mF7vAsNYlOsx9YZO?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| CPT ([Unsloth](https://github.com/unslothai/unsloth)) | Continued Pre-Training using Unsloth for translation. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1gaP8yTle2_v35Um8Gpu9239fqbU7UgY8?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT ([Unsloth](https://github.com/unslothai/unsloth)) | Supervised Fine-Tuning with LoRA using Unsloth. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1vGRg4ksRj__6OLvXkHhvji_Pamv801Ss?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT ([TRL](https://github.com/huggingface/trl)) | Supervised Fine-Tuning with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| DPO ([TRL](https://github.com/huggingface/trl)) | Direct Preference Optimization with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| GRPO ([Unsloth](https://github.com/unslothai/unsloth)) | GRPO with LoRA using Unsloth. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1mIikXFaGvcW4vXOZXLbVTxfBRw_XsXa5?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| GRPO ([TRL](https://github.com/huggingface/trl)) | GRPO with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/github/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_for_verifiable_tasks.ipynb"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |

## 📊 Performance

### Benchmarks

| Model | GPQA Diamond | MMLU-Pro | IFEval | IFBench | Multi-IF |
|---|---|---|---|---|---|
| **LFM2.5-230M** | 25.41 | 20.25 | 71.71 | 38.40 | 37.70 |
| LFM2.5-350M | 30.64 | 20.01 | 76.96 | 40.69 | 44.92 |
| LFM2-350M | 27.58 | 19.29 | 64.96 | 18.20 | 32.92 |
| Granite 4.0-H-350M | 22.32 | 13.14 | 61.27 | 17.22 | 28.70 |
| Granite 4.0-350M | 25.91 | 12.84 | 53.48 | 15.98 | 24.21 |
| Qwen3.5-0.8B (Instruct) | 27.41 | 37.42 | 59.94 | 22.87 | 41.68 |
| Gemma 3 1B IT | 23.89 | 14.04 | 63.49 | 20.33 | 44.25 |

| Model | CaseReportBench | BFCLv3 | BFCLv4 | τ²-Bench Telecom | τ²-Bench Retail |
|---|---|---|---|---|---|
| **LFM2.5-230M** | 22.51 | 43.26 | 21.03 | 5.26 | 13.68 |
| LFM2.5-350M | 32.45 | 44.11 | 21.86 | 18.86 | 17.84 |
| LFM2-350M | 11.67 | 22.95 | 12.29 | 10.82 | 5.56 |
| Granite 4.0-H-350M | 12.44 | 43.07 | 13.28 | 13.74 | 6.14 |
| Granite 4.0-350M | 0.84 | 39.58 | 13.73 | 2.92 | 6.14 |
| Qwen3.5-0.8B (Instruct) | 13.83 | 35.08 | 18.70 | 12.57 | 6.14 |
| Gemma 3 1B IT | 2.28 | 16.61 | 7.17 | 9.36 | 6.43 |

### CPU Inference

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/TCR-MfPtX3YTPvRzxWcG3.png)

### GPU Inference

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/emlcz4gf2wendPhKQWEBN.png)

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI2026230M,
  author = {Liquid AI},
  title = {LFM2.5-230M: Built to Run Anywhere},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2-5-230m},
}
```
```bibtex
@article{liquidai2025lfm2,
  title={LFM2 Technical Report},
  author={Liquid AI},
  journal={arXiv preprint arXiv:2511.23404},
  year={2025}
}
```