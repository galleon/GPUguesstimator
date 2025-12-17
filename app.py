import gradio as gr
import yaml
import math
import matplotlib.pyplot as plt
import os
import json
from huggingface_hub import hf_hub_download

# --- Configuration & Constants ---
HARDWARE_FILE = "hardware_data.yaml"
MODELS_FILE = "models.yaml"

# Physics Constants
COMPUTE_EFFICIENCY = 0.45
MEMORY_EFFICIENCY = 0.70
INTERCONNECT_EFFICIENCY = 0.65


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
            self.hidden_size = self.config.get("hidden_size", 4096)
            self.num_layers = self.config.get("num_hidden_layers", 32)
            self.num_heads = self.config.get("num_attention_heads", 32)
            self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
            self.vocab_size = self.config.get("vocab_size", 32000)
            self.max_context = self.config.get("max_position_embeddings", 4096)
            self.intermediate_size = self.config.get(
                "intermediate_size", self.hidden_size * 4
            )

            self.is_moe = False
            self.num_experts = 1
            self.active_experts = 1

            if "num_local_experts" in self.config:
                self.is_moe = True
                self.num_experts = self.config["num_local_experts"]
                self.active_experts = self.config.get("num_experts_per_tok", 2)
            elif "notes" in self.config and "moe" in self.config["notes"]:
                moe_cfg = self.config["notes"]["moe"]
                self.is_moe = True
                self.num_experts = moe_cfg.get("num_local_experts", 8)
                self.active_experts = moe_cfg.get("num_experts_per_tok", 2)

            self.calculate_params()
        except Exception as e:
            self.error = f"Error parsing config: {str(e)}"

    def calculate_params(self):
        self.params_embed = self.vocab_size * self.hidden_size
        head_dim = self.hidden_size // self.num_heads
        kv_dim = head_dim * self.num_kv_heads

        self.params_attn = (
            (self.hidden_size * self.hidden_size)
            + (self.hidden_size * kv_dim)
            + (self.hidden_size * kv_dim)
            + (self.hidden_size * self.hidden_size)
        )

        dense_mlp = 3 * self.hidden_size * self.intermediate_size

        if self.is_moe:
            self.params_mlp_total = dense_mlp * self.num_experts
            self.params_mlp_active = dense_mlp * self.active_experts
        else:
            self.params_mlp_total = dense_mlp
            self.params_mlp_active = dense_mlp

        self.params_norm = 2 * self.hidden_size
        self.params_layer_total = (
            self.params_attn + self.params_mlp_total + self.params_norm
        )
        self.params_layer_active = (
            self.params_attn + self.params_mlp_active + self.params_norm
        )

        self.total_params = self.params_embed + (
            self.num_layers * self.params_layer_total
        )
        self.active_params = self.params_embed + (
            self.num_layers * self.params_layer_active
        )


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
):
    analyzer = ModelAnalyzer(model_name_or_repo, hf_token)
    if analyzer.error:
        return error_result(analyzer.error)

    if gpu_name not in HARDWARE_DB:
        return error_result(f"GPU '{gpu_name}' not found in database.")

    gpu_spec = HARDWARE_DB[gpu_name]

    # --- Robust Bandwidth Lookup ---
    nvlink_bw = gpu_spec.get("interconnect_bw_gb_s", 0)
    pcie_bw = gpu_spec.get("pcie_bw_gb_s", 64)

    if connectivity_type == "NVLink":
        interconnect_bw = nvlink_bw
        if interconnect_bw == 0:
            return error_result(f"Error: {gpu_name} does not support NVLink.")
    elif connectivity_type == "PCIe / Standard":
        interconnect_bw = pcie_bw
    else:  # Auto
        interconnect_bw = nvlink_bw if nvlink_bw > 0 else pcie_bw

    interconnect_bw_effective = interconnect_bw * INTERCONNECT_EFFICIENCY * 1e9

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

    # --- Memory Calculations ---
    mem_weights = analyzer.total_params * bytes_per_param

    head_dim = analyzer.hidden_size // analyzer.num_heads
    total_tokens = context_in + context_out
    # KV Cache: 2 (K+V) * layers * kv_heads * head_dim * tokens * batch * bytes(2 for FP16)
    mem_kv = (
        2
        * analyzer.num_layers
        * analyzer.num_kv_heads
        * head_dim
        * total_tokens
        * concurrent_users
        * 2
    )

    # Overhead: Reverted to simple 20% rule
    mem_overhead = mem_weights * 0.20

    total_mem_required = mem_weights + mem_kv + mem_overhead
    gpu_mem_capacity = gpu_spec["memory_gb"] * (1024**3)

    num_gpus = math.ceil(total_mem_required / gpu_mem_capacity)

    # --- Latency & Physics ---
    compute_mode = "fp16_tflops_dense"
    total_compute_flops = (
        gpu_spec.get(compute_mode, 100) * 1e12 * num_gpus * COMPUTE_EFFICIENCY
    )
    if quantization == "FP4":
        total_compute_flops *= 2.5

    total_mem_bw = (
        gpu_spec.get("bandwidth_gb_s", 1000) * 1e9 * num_gpus * MEMORY_EFFICIENCY
    )

    # TTFT (Prefill)
    prefill_ops = 2 * analyzer.active_params * context_in * concurrent_users
    time_compute_prefill = prefill_ops / total_compute_flops
    # Move weights + write KV
    time_mem_prefill = (
        mem_weights + (mem_kv * (context_in / total_tokens))
    ) / total_mem_bw
    ttft = max(time_compute_prefill, time_mem_prefill) + (0.05 * num_gpus)

    # TPOT (Decode)
    gen_ops = 2 * analyzer.active_params * concurrent_users
    t_compute = gen_ops / total_compute_flops

    # Load all weights + active KV
    bytes_moved = mem_weights + mem_kv
    t_memory = bytes_moved / total_mem_bw

    # Comm (AllReduce)
    if num_gpus > 1:
        comm_data_per_layer = (
            2 * analyzer.hidden_size * concurrent_users * bytes_per_param
        )
        total_comm_data = comm_data_per_layer * analyzer.num_layers
        t_comm = total_comm_data / interconnect_bw_effective
    else:
        t_comm = 0

    itl = max(t_compute, t_memory) + t_comm

    # --- Result Formatting ---
    server_name = gpu_spec.get("recommended_server", "Contact Lenovo Support")
    if num_gpus > 8:
        server_name += " (Requires Multi-Node Clustering)"

    warnings = []
    if interconnect_bw < 100 and num_gpus > 1:
        warnings.append(
            "Warning: PCIe Bottleneck - High latency expected without NVLink."
        )
    if itl > 0.150:
        warnings.append(
            f"Warning: High Latency - ITL is {itl * 1000:.0f}ms (exceeds 150ms threshold)."
        )
    if analyzer.is_moe:
        warnings.append(
            f"Info: MoE Model - Using active params {analyzer.active_params / 1e9:.1f}B for compute estimates."
        )

    # Chart (Per GPU)
    fig = create_mem_chart_per_gpu(
        mem_weights, mem_kv, mem_overhead, gpu_mem_capacity, num_gpus
    )

    # Textual memory breakdown for accessibility (WCAG 1.1.1 - Text Alternatives)
    w_per_gb = (mem_weights / num_gpus) / (1024**3)
    k_per_gb = (mem_kv / num_gpus) / (1024**3)
    o_per_gb = (mem_overhead / num_gpus) / (1024**3)
    cap_gb = gpu_mem_capacity / (1024**3)
    used_gb = w_per_gb + k_per_gb + o_per_gb
    free_gb = max(0, cap_gb - used_gb)
    total_used_pct = (used_gb / cap_gb * 100) if cap_gb > 0 else 0

    mem_text_alt = (
        f"Per-GPU Memory Breakdown: Weights {w_per_gb:.1f} GB ({w_per_gb / cap_gb * 100:.1f}%), "
        f"KV Cache {k_per_gb:.1f} GB ({k_per_gb / cap_gb * 100:.1f}%), "
        f"Overhead {o_per_gb:.1f} GB ({o_per_gb / cap_gb * 100:.1f}%), "
        f"Free {free_gb:.1f} GB ({free_gb / cap_gb * 100:.1f}%). "
        f"Total used: {used_gb:.1f} GB of {cap_gb:.0f} GB ({total_used_pct:.1f}%)."
    )

    return (
        f"{analyzer.total_params / 1e9:.1f}B",
        f"{total_mem_required / (1024**3):.1f} GB",
        num_gpus,
        f"{ttft * 1000:.0f} ms",
        f"{itl * 1000:.0f} ms",
        server_name,
        "\n".join(warnings) if warnings else "No warnings.",
        fig,
        mem_text_alt,
    )


def create_mem_chart_per_gpu(weights, kv, overhead, single_gpu_cap, num_gpus):
    # Normalize to Per-GPU view
    w_per = (weights / num_gpus) / (1024**3)
    k_per = (kv / num_gpus) / (1024**3)
    o_per = (overhead / num_gpus) / (1024**3)
    cap_gb = single_gpu_cap / (1024**3)

    used = w_per + k_per + o_per
    free = max(0, cap_gb - used)

    # WCAG AA compliant colors with high contrast
    # Using colors that work well with both light and dark backgrounds
    labels = ["Weights", "KV Cache", "Overhead", "Free (Per GPU)"]
    sizes = [w_per, k_per, o_per, free]
    # High contrast colors: blue, purple, orange, gray
    colors = ["#2563eb", "#7c3aed", "#ea580c", "#6b7280"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Enhanced labels with both percentage and GB values for clarity
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = pct * total / 100.0
            return f"{pct:.1f}%\n({val:.1f} GB)" if val > 0.1 else ""

        return my_autopct

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=make_autopct(sizes),
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10, "weight": "bold"},
    )

    # Ensure text is readable (WCAG contrast)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")

    ax.set_title(
        f"Per-GPU Memory Usage (Capacity: {cap_gb:.0f} GB)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax.axis("equal")
    plt.tight_layout()
    plt.close(fig)
    return fig


def error_result(msg):
    empty_fig = plt.figure()
    plt.close(empty_fig)
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
with gr.Blocks(title="GPUguesstimator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPUguesstimator

        Physics-based sizing tool for calculating VRAM requirements, compute capacity, and interconnect bottlenecks for Large Language Model inference.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 1. Workload Configuration")
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
                value=10,
                step=1,
                label="Concurrent Users",
                info="Number of simultaneous inference requests to handle",
            )
            ctx_in = gr.Slider(
                128,
                128000,
                value=2048,
                step=128,
                label="Input Context Length (Tokens)",
                info="Maximum number of input tokens per request",
            )
            ctx_out = gr.Slider(
                128,
                16384,
                value=512,
                step=128,
                label="Output Tokens (Generation Length)",
                info="Maximum number of tokens to generate per request",
            )

            gr.Markdown("## 2. Infrastructure Configuration")
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

            btn = gr.Button("Calculate Sizing", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("## 3. Sizing Results")
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
                        label="TTFT - Time to First Token",
                        info="Prefill latency: time to process input and generate first token",
                    )
                    res_itl = gr.Textbox(
                        label="ITL - Inter-Token Latency",
                        info="Generation speed: time between each generated token",
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
                    lines=2,
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
    demo.launch()
