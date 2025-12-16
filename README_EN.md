# 🥋 PyTorch Black Belt

> **From `import torch` to Engineering Mastery.**
> A comprehensive collection of notebooks designed to bridge the gap between "running a model" and understanding the engine underneath.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Status](https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge)

## 🎯 The Goal

Most tutorials stop at `model.fit()`. This repository starts where they end.
**PyTorch Black Belt** is about understanding the "why" and "how":

- Why does `.view()` throw errors on non-contiguous tensors?
- How do you write a custom Backward pass for a new operation?
- How to debug silent failures that don't throw exceptions but ruin training?
- How to optimize data pipelines when GPU utilization is low?

## 📚 Curriculum: The Path to Mastery

### 🏗️ Module 1: Tensors Deep Dive (Foundations)

_You can't master the model if you don't master the data structure._

- **Broadcasting Dojo:** The math behind dimension expansion.
- **Strides & Memory:** Understanding `storage()`, `view()` vs `reshape()`, and `contiguous()`.
- **Einsum:** The magical function to replace complex matrix ops.
- **In-place Operations:** When `x += 1` saves memory and when it breaks Autograd.

### 🧮 Module 2: Autograd Internals (The Engine)

_Hacking the derivative engine._

- **Computational Graph:** Visualizing dynamic graph construction.
- **`retain_graph=True`:** Use cases beyond the basics.
- **Custom Autograd Functions:** Writing manual `forward` and `backward` methods.
- **Gradient Accumulation:** Simulating large batches on small VRAM.

### 💿 Module 3: Data Engineering (Fuel)

_Garbage In, Garbage Out. Slow In, Slow Training._

- **IterableDataset:** Handling streams and datasets larger than RAM.
- **Custom Collate:** Managing variable-length sequences and padding on the fly.
- **Advanced Sampling:** Balancing imbalanced datasets dynamically.
- **Bottleneck Analysis:** Optimizing `num_workers`, `pin_memory`, and prefetching.

### 🧠 Module 4: Advanced Architecture (Construction)

_Building robust and complex systems._

- **Hooks:** Inspecting activations and gradients inside the black box.
- **Initialization Strategies:** Why Xavier and Kaiming matter for convergence.
- **Weight Sharing:** Tying parameters between layers (e.g., Autoencoders).
- **Dynamic Control Flow:** Using Python logic (`if/else`) inside the graph.

### ⚡ Module 5: Training & Optimization (Speed)

_Squeezing every FLOPS out of your GPU._

- **Mixed Precision (AMP):** implementing `fp16` for 2x speedup.
- **Schedulers:** Warmup, Cosine Annealing, and Cyclic Learning Rates.
- **Gradient Clipping:** Preventing exploding gradients in RNNs/Transformers.
- **Torch 2.0:** Mastering `torch.compile` and fusion strategies.

### 📦 Module 6: Ecosystem & Production (Scale)

_Moving from notebook to cluster._

- **PyTorch Lightning:** Structuring code for reproducibility.
- **TorchScript & Tracing:** Exporting models to C++ environments.
- **DDP (Distributed Data Parallel):** Multi-GPU training mechanics.
- **Profiling:** Using PyTorch Profiler to find code bottlenecks.

## 🛠️ Tech Stack & Tools

This project focuses on the modern, high-performance PyTorch ecosystem:

- **Python 3.10+**
- **PyTorch 2.x** (Core framework, focusing on `torch.compile` and dynamic graphs)
- **Einops** (Readable and powerful tensor operations)
- **PyTorch Lightning** (Organizing complex training pipelines)
- **Torch Profiler & TensorBoard** (Performance debugging)
- **NumPy & Pandas** (Data manipulation)
- **Matplotlib & Seaborn** (Visualizing internals and loss landscapes)

## 🚀 How to Use

You have two options: instant cloud execution or a professional local setup.

### ☁️ Option 1: Google Colab (Zero Setup)

The fastest way to learn. Every notebook in this repository has an **"Open in Colab"** badge at the top.

1.  Open any `.ipynb` file in the file list.
2.  Click the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle"> badge.
3.  The code runs immediately on Google's free GPUs.

### 💻 Option 2: Local Development (VS Code + uv)

Recommended for engineers building their own experimental environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/takzen/pytorch-black-belt.git
    cd pytorch-black-belt
    ```

2.  **Initialize environment with `uv`:**

    ```bash
    # Create virtual environment
    uv venv

    # Activate it:
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install Dependencies (The Engineering Stack):**

    ```bash
    # 1. Install PyTorch with CUDA support (Adjust index-url for your GPU)
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

    # 2. Install Tools (Einops, Profilers, Visualization)
    uv pip install numpy pandas matplotlib seaborn einops lightning tensorboard torch-tb-profiler jupyterlab ipywidgets
    ```

---

## 📊 Project Stats

- **Comprehensive curriculum** focusing purely on PyTorch internals and engineering.
- **From Math to Production:** From manual backpropagation implementation to distributed training (DDP).
- **6 Deep Dive Modules:** Tensors, Autograd, Data Engineering, Architecture, Optimization, Production.
- **Reference Implementations:** Custom CUDA kernels, Memory-efficient Attention, Gradient Checkpointing.
- **Modern PyTorch 2.0:** Leveraging `torch.compile` and Fusion strategies.

---

**Author:** Krzysztof Pika
