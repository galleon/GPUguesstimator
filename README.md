---
title: GPUguesstimator
emoji: 🌍
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
license: apache-2.0
---

# GPUguesstimator

Physics-based sizing tool for LLM inference. Given a model, a GPU, and a workload description, it estimates how many GPUs you need, how much memory they will use, and what latency to expect.

## Quick start

```bash
uv venv
uv pip install -r requirements.txt
uv run python app.py
```

---

## How it works

The tool performs three independent calculations: memory sizing, GPU count, and latency estimation.

### 1. Model parameters

For models listed in `models.yaml`, architecture parameters (layer count, hidden dimension, etc.) are read directly from the file. For any other model, the tool fetches `config.json` from Hugging Face and derives the same values. When the Hugging Face API is reachable, the exact parameter count is read from the model's safetensors metadata; otherwise it is estimated from the architecture using the formula below.

**Parameter count formula (dense models):**

```
embed_params     = vocab_size × hidden_size
attn_params      = 2 × hidden_size² + 2 × hidden_size × (head_dim × num_kv_heads)
mlp_params       = 3 × hidden_size × intermediate_size
total_params     = embed_params + num_layers × (attn_params + mlp_params)
```

For **Mixture-of-Experts (MoE)** models, `mlp_params` is multiplied by the total number of experts. Active parameters (used for compute estimates) are then scaled down by the expert routing ratio (`active_experts / total_experts`).

---

### 2. Memory sizing

Total required memory is split into static and dynamic portions.

**Static memory** — fixed for a given model and precision:

```
weights_memory = total_params × bytes_per_param
rag_memory     = embedding_model_size + reranker_model_size  (if RAG enabled)
static_memory  = weights_memory + rag_memory
```

`bytes_per_param` depends on the selected quantization:

| Precision   | Bytes per parameter | Notes                          |
|-------------|---------------------|--------------------------------|
| FP16 / BF16 | 2                   | Standard inference             |
| INT8        | 1                   | 8-bit quantization             |
| FP4         | 0.5                 | Requires Blackwell GPUs        |

**Dynamic memory** — grows with the number of concurrent users and context length:

```
kv_cache_per_user  = 2 × num_layers × num_kv_heads × head_dim × (input_tokens + output_tokens) × bytes_per_param
activation_per_user = 0.5 GB  (fixed buffer)
dynamic_memory      = (kv_cache_per_user + activation_per_user) × concurrent_users
```

The **KV cache** (Key-Value cache) is the memory used to store the attention keys and values computed during the input processing phase, so they do not need to be recomputed during token generation. Its size scales with context length and the number of attention heads. Models using **Grouped Query Attention (GQA)** — where `num_kv_heads < num_attention_heads` — have a significantly smaller KV cache than full multi-head attention models.

**Total memory and overhead:**

```
total_memory = static_memory + dynamic_memory × (1 + overhead_pct) + 1 GB (CUDA fixed overhead)
```

The overhead percentage accounts for memory fragmentation and runtime allocations. It is applied only to the dynamic portion because static weight tensors have a fixed, known size. A fixed 1 GB constant covers the CUDA driver context and other per-process allocations.

**GPU count:**

```
num_gpus = ceil(total_memory / gpu_vram_capacity)
```

---

### 3. Latency estimation

Two latency metrics are estimated.

**TTFT — Time to First Token**

TTFT measures how long the user waits before seeing the first word of the response. It is dominated by the *prefill* phase, where the model processes the entire input prompt in one forward pass.

```
prefill_flops    = 2 × active_params × input_tokens × concurrent_users
t_compute_prefill = prefill_flops / effective_flops
t_memory_prefill  = weights_memory / effective_memory_bandwidth
TTFT             = max(t_compute_prefill, t_memory_prefill) + rag_processing_latency
```

When RAG (Retrieval-Augmented Generation) is enabled, an additional latency term is added for embedding the query and optionally reranking retrieved documents before the LLM receives its input.

**ITL — Inter-Token Latency**

ITL measures the time between each successive generated token during the decode phase. Each decode step is memory-bandwidth-bound: the GPU must stream the entire weight tensor and all active KV caches from HBM (High Bandwidth Memory) to compute a single output token.

```
decode_flops = 2 × active_params × concurrent_users
t_compute_gen = decode_flops / effective_flops
bytes_per_step = weights_memory + total_kv_cache   (all users)
t_memory_gen  = bytes_per_step / effective_memory_bandwidth
ITL           = max(t_compute_gen, t_memory_gen)
```

**Multi-GPU scaling:**

With NVLink interconnect, compute and bandwidth scale linearly with GPU count. With PCIe, effective memory bandwidth is capped at a single card's bandwidth (the inter-GPU link becomes a bottleneck), and latency penalties are applied.

| Setup          | Effective FLOPS          | Effective bandwidth      |
|----------------|--------------------------|--------------------------|
| Single GPU     | 1× GPU FLOPS             | 1× GPU bandwidth         |
| Multi + NVLink | N× GPU FLOPS             | N× GPU bandwidth         |
| Multi + PCIe   | N× GPU FLOPS             | 1× GPU bandwidth (capped)|

---

### 4. RAG pipeline sizing

When RAG is enabled, two additional models are loaded into VRAM alongside the main LLM:

- **Embedding model** — encodes the user query into a vector for similarity search against the document store.
- **Reranker model** — re-scores the top retrieved documents before they are inserted into the LLM prompt. Optional.

Their combined VRAM is added to the static memory footprint, and their inference latency is added to TTFT.

---

## Configuration files

### `models.yaml`

Stores architecture parameters for bundled model presets. Each entry maps a Hugging Face repository ID to the fields needed for sizing:

```yaml
"meta-llama/Meta-Llama-3-70B-Instruct":
  hidden_size: 8192
  num_hidden_layers: 80
  num_attention_heads: 64
  num_key_value_heads: 8        # GQA: fewer KV heads → smaller KV cache
  vocab_size: 128256
  intermediate_size: 28672
```

MoE models include a `notes.moe` block with `num_local_experts` and `num_experts_per_tok`. Any Hugging Face model ID not present in this file is fetched dynamically.

### `hardware_data.yaml`

Stores GPU specifications. Key fields:

| Field | Description |
|---|---|
| `memory_gb` | VRAM capacity |
| `bandwidth_gb_s` | HBM memory bandwidth (GB/s) — the main bottleneck during decode |
| `fp16_tflops_dense` | FP16 compute throughput — the main factor during prefill |
| `interconnect_bw_gb_s` | NVLink bandwidth; 0 means PCIe only |
| `fp4_supported` | Whether the GPU supports FP4 precision (Blackwell and newer) |

---

## Glossary

| Term | Meaning |
|---|---|
| **VRAM** | Video RAM — the on-chip memory of a GPU |
| **HBM** | High Bandwidth Memory — the memory technology used in datacenter GPUs (H100, A100, etc.) |
| **FLOPS** | Floating Point Operations Per Second — measure of compute throughput |
| **FP16 / BF16** | 16-bit floating point formats used for standard inference |
| **INT8 / FP4** | Reduced-precision formats that cut memory use at the cost of some accuracy |
| **KV cache** | Key-Value cache — stores intermediate attention results to avoid recomputing them during generation |
| **GQA** | Grouped Query Attention — shares key/value heads across query heads to reduce KV cache size |
| **MoE** | Mixture of Experts — architecture where only a subset of the model's layers (experts) are active per token |
| **TTFT** | Time to First Token — latency from sending a request to receiving the first generated token |
| **ITL** | Inter-Token Latency — time between each successive token during generation |
| **Prefill** | The phase where the model processes the input prompt (compute-bound) |
| **Decode** | The phase where the model generates output tokens one by one (memory-bandwidth-bound) |
| **RAG** | Retrieval-Augmented Generation — augments LLM input with documents retrieved from an external store |
| **NVLink** | NVIDIA's high-bandwidth GPU-to-GPU interconnect, used in SXM-form-factor server GPUs |
| **PCIe** | Standard expansion bus used to connect GPUs to the CPU; lower bandwidth than NVLink |
| **Quantization** | Representing model weights in lower precision to reduce memory and increase throughput |
