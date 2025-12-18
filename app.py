import gradio as gr
import yaml
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import json
from huggingface_hub import hf_hub_download, HfApi

# --- Configuration & Constants ---
HARDWARE_FILE = "hardware_data.yaml"
MODELS_FILE = "models.yaml"

# Physics Constants
COMPUTE_EFFICIENCY = 0.45
MEMORY_EFFICIENCY = 0.70
INTERCONNECT_EFFICIENCY = 0.65

# Defaults
ACTIVATION_MEMORY_BUFFER_GB = 0.5
DEFAULT_GPU_OVERHEAD_PCT = 20

# Embedding Models VRAM Est. (Weights + Runtime Buffer)
EMBEDDING_MODELS = {
    "External/API (No Local VRAM)": 0.0,
    "Mini (All-MiniLM-L6) ~0.2GB": 0.2,
    "Standard (MPNet-Base/BGE-Base) ~0.6GB": 0.6,
    "Large (BGE-M3/GTE-Large) ~2.5GB": 2.5,
    "LLM-Based (E5-Mistral-7B) ~16GB": 16.0,
}

# Reranker Models VRAM Est. (Weights + Batch Processing Buffer)
RERANKER_MODELS = {
    "None (Skip Reranking)": 0.0,
    "Small (BGE-Reranker-Base) ~0.5GB": 0.5,
    "Large (BGE-Reranker-Large) ~1.5GB": 1.5,
    "LLM-Based (BGE-Reranker-v2-Gemma) ~10GB": 10.0,
}


# --- Data Loading ---
def load_hardware_data():
    if not os.path.exists(HARDWARE_FILE):
        return {}
    with open(HARDWARE_FILE, "r") as f:
        data = yaml.safe_load(f)
    return {gpu["name"]: gpu for gpu in data["gpus"]}


def load_models_data():
    if not os.path.exists(MODELS_FILE):
        return {}
    with open(MODELS_FILE, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


HARDWARE_DB = load_hardware_data()
MODELS_DB = load_models_data()


# --- Model Analysis ---
class ModelAnalyzer:
    def __init__(self, repo_id, hf_token=None):
        self.repo_id = repo_id
        self.config = {}
        self.error = None
        self.api = HfApi(token=hf_token.strip() if hf_token else None)

        # 1. Try to get Model Info (Total Params) from API first
        self.total_params_safetensors = None
        try:
            model_info = self.api.model_info(repo_id)
            if hasattr(model_info, "safetensors") and model_info.safetensors and "total" in model_info.safetensors:
                self.total_params_safetensors = model_info.safetensors["total"]
        except Exception:
            pass  # Fallback to config parsing

        # 2. Load Config
        if repo_id in MODELS_DB:
            self.config = MODELS_DB[repo_id]
        else:
            try:
                token = hf_token.strip() if hf_token else None
                config_path = hf_hub_download(
                    repo_id=repo_id, filename="config.json", token=token
                )
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                self.error = f"Failed to fetch model: {str(e)}"
                return

        try:
            # Handle nested configs (common in multimodal)
            if "text_config" in self.config:
                self.llm_config = self.config["text_config"]
            elif "llm_config" in self.config:
                self.llm_config = self.config["llm_config"]
            else:
                self.llm_config = self.config

            self.hidden_size = self.llm_config.get("hidden_size", 4096)
            self.num_layers = self.llm_config.get("num_hidden_layers", 32)
            self.num_heads = self.llm_config.get("num_attention_heads", 32)
            self.num_kv_heads = self.llm_config.get("num_key_value_heads", self.num_heads)
            self.vocab_size = self.llm_config.get("vocab_size", 32000)
            self.max_context = self.llm_config.get("max_position_embeddings", 4096)
            self.intermediate_size = self.llm_config.get(
                "intermediate_size", self.hidden_size * 4
            )

            # MoE detection
            self.is_moe = False
            self.num_experts = 1
            self.active_experts = 1

            # Check for MoE config patterns
            self._detect_moe()

            # Calculate Parameters
            self.calculate_params()

        except Exception as e:
            self.error = f"Error parsing config: {str(e)}"

    def _detect_moe(self):
        archs = self.config.get("architectures", [])
        keys = set(self.config.keys()) | set(self.llm_config.keys())

        if (
            any("moe" in a.lower() for a in archs)
            or any("moe" in k.lower() for k in keys)
            or any("expert" in k.lower() for k in keys)
        ):
            self.is_moe = True

        if self.is_moe:
            self.num_experts = (
                self.llm_config.get("num_local_experts")
                or self.llm_config.get("num_experts")
                or self.llm_config.get("n_routed_experts")
                or 8
            )
            self.active_experts = (
                self.llm_config.get("num_experts_per_tok")
                or self.llm_config.get("num_experts_per_token")
                or 2
            )
        elif "notes" in self.config and "moe" in self.config["notes"]:
            moe_cfg = self.config["notes"]["moe"]
            self.is_moe = True
            self.num_experts = moe_cfg.get("num_local_experts", 8)
            self.active_experts = moe_cfg.get("num_experts_per_tok", 2)

    def calculate_params(self):
        # If we got exact params from safetensors, use that
        if self.total_params_safetensors:
            self.total_params = self.total_params_safetensors
        else:
            # Fallback calculation
            self.params_embed = self.vocab_size * self.hidden_size
            head_dim = self.hidden_size // self.num_heads
            kv_dim = head_dim * self.num_kv_heads

            self.params_attn = (
                (self.hidden_size * self.hidden_size)
                + (self.hidden_size * kv_dim) * 2
                + (self.hidden_size * self.hidden_size)
            )
            dense_mlp = 3 * self.hidden_size * self.intermediate_size

            if self.is_moe:
                mlp_total = dense_mlp * self.num_experts
            else:
                mlp_total = dense_mlp

            self.params_norm = 2 * self.hidden_size
            self.params_layer_total = (
                self.params_attn + mlp_total + self.params_norm
            )
            self.total_params = self.params_embed + (
                self.num_layers * self.params_layer_total
            )

        # Active Params Calculation (using improved heuristic for MoE)
        if self.is_moe:
            expert_param_fraction = 0.8  # 80% of params are in experts
            always_active = self.total_params * (1 - expert_param_fraction)
            expert_params = self.total_params * expert_param_fraction
            expert_ratio = self.active_experts / self.num_experts
            self.active_params = int(
                always_active + (expert_params * expert_ratio)
            )
        else:
            self.active_params = self.total_params


# --- Calculation Engine ---
def calculate_dimensioning(
    model_name_or_repo,
    hf_token,
    gpu_name,
    connectivity_type,
    concurrent_users,
    context_in,
    context_out,
    quantization,
    gpu_overhead_pct,
    rag_enabled,
    rag_model_key,
    reranker_model_key,
):
    analyzer = ModelAnalyzer(model_name_or_repo, hf_token)
    if analyzer.error:
        return error_result(analyzer.error)

    if gpu_name not in HARDWARE_DB:
        return error_result(f"GPU '{gpu_name}' not found in database.")

    gpu_spec = HARDWARE_DB[gpu_name]

    # 2. Interconnect & Bandwidth Logic
    nvlink_bw = gpu_spec.get("interconnect_bw_gb_s", 0)
    pcie_bw = gpu_spec.get("pcie_bw_gb_s", 64)
    gpu_has_nvlink = nvlink_bw > 0

    if connectivity_type == "NVLink":
        if not gpu_has_nvlink:
            return error_result(f"Error: {gpu_name} does not support NVLink.")
        using_nvlink = True
        interconnect_bw_effective = nvlink_bw * INTERCONNECT_EFFICIENCY * 1e9
    elif connectivity_type == "PCIe / Standard":
        using_nvlink = False
        interconnect_bw_effective = pcie_bw * 1e9  # PCIe usually raw
    else:  # Auto
        using_nvlink = gpu_has_nvlink
        interconnect_bw_effective = (
            (nvlink_bw if using_nvlink else pcie_bw) * 1e9
        )

    # --- Precision ---
    fp4_supported = gpu_spec.get("fp4_supported", False)

    if quantization == "FP16/BF16":
        bytes_per_param = 2
    elif quantization == "INT8":
        bytes_per_param = 1
    elif quantization == "FP4":
        if not fp4_supported:
            return error_result(f"Error: {gpu_name} does not support FP4.")
        bytes_per_param = 0.5
    else:
        bytes_per_param = 2

    # --- MEMORY CALCULATION ---

    # Static Footprint
    mem_weights = analyzer.total_params * bytes_per_param

    # RAG Memory (Embedding + Reranker)
    mem_rag = 0
    if rag_enabled:
        embed_gb = EMBEDDING_MODELS.get(rag_model_key, 0.6)
        rerank_gb = RERANKER_MODELS.get(reranker_model_key, 0.5)
        mem_rag = (embed_gb + rerank_gb) * (1024**3)

    static_footprint = mem_weights + mem_rag

    # Dynamic Footprint (KV + Activation per user)
    head_dim = analyzer.hidden_size // analyzer.num_heads
    total_tokens = context_in + context_out

    # KV Cache
    kv_bytes = 2
    mem_kv_per_user = (
        2
        * analyzer.num_layers
        * analyzer.num_kv_heads
        * head_dim
        * total_tokens
        * kv_bytes
    )

    # Activation buffer
    mem_act_per_user = ACTIVATION_MEMORY_BUFFER_GB * 1024**3

    dynamic_per_user = mem_kv_per_user + mem_act_per_user
    total_dynamic = dynamic_per_user * concurrent_users

    # Total & Overhead
    raw_total_mem = static_footprint + total_dynamic
    total_mem_required = raw_total_mem * (1 + gpu_overhead_pct / 100)

    gpu_mem_capacity = gpu_spec["memory_gb"] * (1024**3)
    num_gpus = math.ceil(total_mem_required / gpu_mem_capacity)

    # --- LATENCY CALCULATION ---
    compute_mode = "fp16_tflops_dense"
    single_gpu_flops = (
        gpu_spec.get(compute_mode, 100) * 1e12 * COMPUTE_EFFICIENCY
    )
    if quantization == "FP4":
        single_gpu_flops *= 2.5

    single_gpu_bw = (
        gpu_spec.get("bandwidth_gb_s", 1000) * 1e9 * MEMORY_EFFICIENCY
    )

    if num_gpus == 1:
        effective_flops = single_gpu_flops
        effective_mem_bw = single_gpu_bw
        ttft_penalty = 2.0
        itl_penalty = 1.0
    elif using_nvlink:
        effective_flops = single_gpu_flops * num_gpus
        effective_mem_bw = single_gpu_bw * num_gpus
        ttft_penalty = 2.0
        itl_penalty = 1.0
    else:
        # PCIe Bottleneck Logic
        effective_flops = single_gpu_flops * num_gpus
        effective_mem_bw = single_gpu_bw  # Capped at single card
        n = num_gpus
        ttft_penalty = 1.2 * n * n - n
        itl_penalty = n

    # TTFT (Prefill) + RAG Latency

    # 1. RAG Processing (Embedding + Reranking)
    t_rag_processing = 0
    if rag_enabled:
        # Base Embedding Latency (Encode Query)
        if "Mini" in rag_model_key:
            t_rag_processing += 0.02
        elif "Large" in rag_model_key:
            t_rag_processing += 0.05
        elif "LLM" in rag_model_key:
            t_rag_processing += 0.15
        else:
            t_rag_processing += 0.03

        # Reranking Latency (Process Documents)
        if "None" not in reranker_model_key:
            if "Small" in reranker_model_key:
                t_rag_processing += 0.15  # 150ms
            elif "Large" in reranker_model_key:
                t_rag_processing += 0.35  # 350ms
            elif "LLM" in reranker_model_key:
                t_rag_processing += 0.80  # 800ms

    # 2. LLM Compute Time
    prefill_ops = 2 * analyzer.active_params * context_in * concurrent_users
    t_compute_prefill = (prefill_ops / effective_flops) * ttft_penalty
    t_mem_prefill = mem_weights / effective_mem_bw

    ttft = max(t_compute_prefill, t_mem_prefill) + t_rag_processing

    # ITL (Decode)
    gen_ops = 2 * analyzer.active_params * concurrent_users
    t_compute_gen = (gen_ops / effective_flops) * itl_penalty
    bytes_per_step = mem_weights + (total_dynamic / concurrent_users)
    t_mem_gen = (bytes_per_step / effective_mem_bw) * itl_penalty
    itl = max(t_compute_gen, t_mem_gen)

    # --- Result Formatting ---
    server_name = gpu_spec.get("recommended_server", "Contact Lenovo Support")
    if num_gpus > 8:
        server_name += " (Requires Multi-Node Clustering)"

    warnings = []
    if not using_nvlink and num_gpus > 1:
        warnings.append(
            f"⚠️ No NVLink: Effective Bandwidth capped at {gpu_spec['bandwidth_gb_s']} GB/s. High latency penalty."
        )
    if itl > 0.150:
        warnings.append(
            f"⚠️ High Latency: ITL is {itl * 1000:.0f}ms (>150ms)."
        )
    if t_rag_processing > 0.5:
        warnings.append(
            f"⚠️ High RAG Latency: Reranking is adding {t_rag_processing * 1000:.0f}ms to TTFT."
        )
    if analyzer.is_moe:
        warnings.append(
            f"ℹ️ MoE Model: Active params {analyzer.active_params / 1e9:.1f}B used for compute."
        )
    if rag_enabled:
        warnings.append(
            f"ℹ️ RAG Enabled: Allocating {mem_rag / (1024**3):.1f}GB for Models (Embed+Rerank)."
        )

    # Chart (Per GPU)
    overhead_bytes = raw_total_mem * (gpu_overhead_pct / 100)
    fig = create_mem_chart_per_gpu(
        mem_weights,
        mem_rag,
        total_dynamic,
        overhead_bytes,
        gpu_mem_capacity,
        num_gpus,
    )

    # Textual memory breakdown for accessibility (WCAG 1.1.1 - Text Alternatives)
    w_per_gb = (mem_weights / num_gpus) / (1024**3)
    r_per_gb = (mem_rag / num_gpus) / (1024**3)
    d_per_gb = (total_dynamic / num_gpus) / (1024**3)
    o_per_gb = (overhead_bytes / num_gpus) / (1024**3)
    cap_gb = gpu_mem_capacity / (1024**3)
    used_gb = w_per_gb + r_per_gb + d_per_gb + o_per_gb
    free_gb = max(0, cap_gb - used_gb)
    total_used_pct = (used_gb / cap_gb * 100) if cap_gb > 0 else 0

    # Calculate percentages for display
    w_pct = (w_per_gb / cap_gb * 100) if cap_gb > 0 else 0
    r_pct = (r_per_gb / cap_gb * 100) if cap_gb > 0 else 0
    d_pct = (d_per_gb / cap_gb * 100) if cap_gb > 0 else 0
    o_pct = (o_per_gb / cap_gb * 100) if cap_gb > 0 else 0
    free_pct = (free_gb / cap_gb * 100) if cap_gb > 0 else 0

    mem_text_alt = (
        f"Per-GPU Memory Breakdown (Total Capacity: {cap_gb:.0f} GB):\n"
        f"• Weights: {w_per_gb:.1f} GB ({w_pct:.1f}%) - Model parameters stored in memory. Fixed size based on model architecture and quantization.\n"
        f"• RAG Models: {r_per_gb:.1f} GB ({r_pct:.1f}%) - Embedding and reranker models. Only allocated if RAG is enabled.\n"
        f"• Dynamic (KV+Act): {d_per_gb:.1f} GB ({d_pct:.1f}%) - KV cache and activation buffers. Grows with concurrent users, input context length, and output tokens.\n"
        f"• Overhead: {o_per_gb:.1f} GB ({o_pct:.1f}%) - CUDA context, memory fragmentation, and system buffers. Configurable percentage of total memory.\n"
        f"• Free: {free_gb:.1f} GB ({free_pct:.1f}%) - Available memory headroom for additional operations."
    )

    return (
        f"{analyzer.total_params / 1e9:.1f}B (Active: {analyzer.active_params / 1e9:.1f}B)",
        f"{total_mem_required / (1024**3):.1f} GB",
        num_gpus,
        f"{ttft * 1000:.0f} ms",
        f"{itl * 1000:.0f} ms",
        server_name,
        "\n".join(warnings) if warnings else "No warnings.",
        fig,
        mem_text_alt,
    )


def create_mem_chart_per_gpu(
    weights, rag, dynamic, overhead, single_gpu_cap, num_gpus
):
    # Normalize to Per-GPU view
    w_per = (weights / num_gpus) / (1024**3)
    r_per = (rag / num_gpus) / (1024**3)
    d_per = (dynamic / num_gpus) / (1024**3)
    o_per = (overhead / num_gpus) / (1024**3)
    cap_gb = single_gpu_cap / (1024**3)

    used = w_per + r_per + d_per + o_per
    free = max(0, cap_gb - used)

    # Modern, accessible color palette (WCAG AA compliant)
    labels = ["Weights", "RAG Models", "Dynamic (KV+Act)", "Overhead", "Free (Per GPU)"]
    values = [w_per, r_per, d_per, o_per, free]

    # Filter out zero values for cleaner chart
    clean_labels = []
    clean_values = []
    colors_full = ["#4A90E2", "#10b981", "#8b5cf6", "#f59e0b", "#BDC3C7"]
    clean_colors = []

    for i, val in enumerate(values):
        if val > 0.05:  # Only show if > 50MB
            clean_labels.append(labels[i])
            clean_values.append(val)
            clean_colors.append(colors_full[i])

    # Professional color palette: Blue, Green, Purple, Orange, Gray
    colors = clean_colors if clean_colors else colors_full[: len(clean_values)]

    # Calculate percentages for hover text
    total = sum(clean_values) if clean_values else sum(values)
    percentages = [
        (v / total * 100) if total > 0 else 0
        for v in (clean_values if clean_values else values)
    ]

    # Create hover text with detailed information
    display_labels = clean_labels if clean_labels else labels
    display_values = clean_values if clean_values else values
    hover_texts = [
        f"{display_labels[i]}<br>"
        f"Value: {display_values[i]:.1f} GB<br>"
        f"Percentage: {percentages[i]:.1f}%<br>"
        f"Capacity: {cap_gb:.0f} GB"
        for i in range(len(display_labels))
    ]

    # Create donut chart using plotly
    fig = go.Figure(
        data=[
            go.Pie(
                labels=display_labels,
                values=display_values,
                hole=0.5,  # Creates the donut (hole in the middle)
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts,
            )
        ]
    )

    # Update layout for better appearance
    fig.update_layout(
        title={
            "text": f"Per-GPU Memory Usage (Capacity: {cap_gb:.0f} GB)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial, sans-serif"},
        },
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
    )

    return fig


def error_result(msg):
    # Create an empty plotly figure for error state
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="Error: Unable to generate chart",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    empty_fig.update_layout(
        title="Memory Breakdown",
        height=500,
        showlegend=False,
    )
    return (
        "Error",
        "Error",
        0,
        "-",
        "-",
        "Check Inputs",
        f"Error: {msg}",
        empty_fig,
        "Memory breakdown not available due to calculation error.",
    )


# --- UI Setup ---
# Custom CSS for better font rendering
custom_css = """
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
"""

with gr.Blocks(title="GPUguesstimator") as demo:
    gr.Markdown(
        """
        # GPUguesstimator

        Physics-based sizing tool for calculating VRAM requirements, compute capacity, and interconnect bottlenecks for Large Language Model inference.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Workload Configuration")
            model_keys = list(MODELS_DB.keys())
            model_dd = gr.Dropdown(
                choices=model_keys + ["Custom"],
                value=model_keys[0] if model_keys else "Custom",
                label="Model Preset",
                info="Select a preset model or choose Custom to enter a HuggingFace repository ID",
            )
            repo_input = gr.Textbox(
                label="HuggingFace Repository ID",
                value=model_keys[0] if model_keys else "",
                placeholder="e.g., meta-llama/Meta-Llama-3-70B-Instruct",
                info="Enter the HuggingFace model repository identifier",
            )
            hf_token = gr.Textbox(
                label="HuggingFace Token (Optional)",
                type="password",
                info="Required for accessing gated models. Leave empty for public models.",
            )

            users = gr.Slider(
                1,
                500,
                value=50,
                step=1,
                label="Concurrent Users",
                info="Number of simultaneous inference requests to handle",
            )
            ctx_in = gr.Slider(
                128,
                128000,
                value=1024,
                step=128,
                label="Input Context Length (Tokens)",
                info="Maximum number of input tokens per request",
            )
            ctx_out = gr.Slider(
                128,
                16384,
                value=256,
                step=128,
                label="Output Tokens (Generation Length)",
                info="Maximum number of tokens to generate per request",
            )

            with gr.Group():
                gr.Markdown("#### Retrieval Augmented Generation (RAG)")
                rag_chk = gr.Checkbox(
                    label="Enable RAG Pipeline", value=False
                )
                with gr.Row():
                    rag_model_dd = gr.Dropdown(
                        choices=list(EMBEDDING_MODELS.keys()),
                        value="Standard (MPNet-Base/BGE-Base) ~0.6GB",
                        label="Embedding Model",
                        interactive=True,
                    )
                    rerank_model_dd = gr.Dropdown(
                        choices=list(RERANKER_MODELS.keys()),
                        value="None (Skip Reranking)",
                        label="Reranker Model",
                        interactive=True,
                    )

            gr.Markdown("## Infrastructure Configuration")
            gpu_keys = list(HARDWARE_DB.keys())
            default_gpu = gpu_keys[0] if gpu_keys else "NVIDIA H100-80GB SXM5"

            gpu_select = gr.Dropdown(
                choices=gpu_keys,
                value=default_gpu,
                label="GPU Model",
                info="Select the GPU model for inference",
            )
            conn_select = gr.Dropdown(
                choices=["Auto", "NVLink", "PCIe / Standard"],
                value="Auto",
                label="Interconnect Type",
                info="Auto uses GPU default, NVLink for high-bandwidth, PCIe for standard connections",
            )
            quant_select = gr.Dropdown(
                choices=["FP16/BF16", "INT8", "FP4"],
                value="FP16/BF16",
                label="Quantization Precision",
                info="Model weight precision: FP16/BF16 (standard), INT8 (8-bit), FP4 (4-bit, requires Blackwell)",
            )
            overhead_slider = gr.Slider(
                0,
                50,
                value=20,
                step=5,
                label="GPU Memory Overhead %",
                info="Additional memory overhead percentage for CUDA context, fragmentation, and system buffers",
            )

            btn = gr.Button("Calculate Sizing", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("## Sizing Results")
            with gr.Group():
                res_gpus = gr.Number(
                    label="GPUs Required",
                    precision=0,
                    info="Minimum number of GPUs needed to fit the model and workload",
                )
                res_server = gr.Textbox(
                    label="Recommended Lenovo Server",
                    info="Suggested Lenovo server configuration",
                )
                res_vram = gr.Textbox(
                    label="Total VRAM Required",
                    info="Total video memory needed across all GPUs",
                )
                res_params = gr.Textbox(
                    label="Model Parameters",
                    info="Total number of model parameters in billions",
                )
                with gr.Row():
                    res_ttft = gr.Textbox(
                        label="TTFT - Time to First Token (Prefill latency)",
                        info="time to process input and generate first token",
                    )
                    res_itl = gr.Textbox(
                        label="ITL - Inter-Token Latency",
                        info="time between each generated token",
                    )
                res_warnings = gr.Textbox(
                    label="Analysis Notes and Warnings",
                    lines=4,
                    info="Important notes, warnings, and recommendations about the configuration",
                )
                plot_output = gr.Plot(label="Per-GPU Memory Breakdown Chart")
                mem_text_alt = gr.Textbox(
                    label="Memory Breakdown (Text Description)",
                    info="Textual description of memory allocation for screen readers and accessibility",
                    lines=6,
                )

    def update_repo(choice):
        return choice if choice != "Custom" else ""

    model_dd.change(update_repo, model_dd, repo_input)

    btn.click(
        calculate_dimensioning,
        inputs=[
            repo_input,
            hf_token,
            gpu_select,
            conn_select,
            users,
            ctx_in,
            ctx_out,
            quant_select,
            overhead_slider,
            rag_chk,
            rag_model_dd,
            rerank_model_dd,
        ],
        outputs=[
            res_params,
            res_vram,
            res_gpus,
            res_ttft,
            res_itl,
            res_server,
            res_warnings,
            plot_output,
            mem_text_alt,
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)
