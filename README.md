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

# LLM GPU Sizer (Gradio)

This Space estimates:
- VRAM for model weights + KV cache (worst-case per concurrency)
- number of GPUs required (with headroom)
- TTFT and ITL (anchor-based simulation)
- optionally reads TTFT/ITL from a running vLLM server `/metrics`

## Local dev (uv)
```bash
uv venv
uv pip install -r requirements.txt
uv run python app.py
