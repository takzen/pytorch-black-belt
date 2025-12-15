# PyTorch Black Belt

## Summary

The 'pytorch-black-belt' project is an advanced educational course designed to teach the deep engineering mechanics of PyTorch. It aims to bridge the gap between basic model usage and true framework mastery, focusing on the "why" behind the framework's design and functionality.

## Project Goal

The core philosophy is to provide a "black belt" level of understanding of PyTorch. This means going beyond surface-level API calls to explore the underlying engineering principles, such as memory management, autograd internals, and performance optimization. The course is structured as a series of Jupyter notebooks, written in Polish, that combine in-depth explanations with practical code examples.

## Curriculum/Roadmap

The full curriculum is detailed in `ROADMAP.md` and consists of 50 planned Jupyter notebooks organized into 7 distinct modules:

1.  **Module 1: Tensor Mechanics** - Low-level details of tensor storage and memory.
2.  **Module 2: The Autograd Engine** - Internals of automatic differentiation.
3.  **Module 3: Advanced APIs** - Deep dives into hooks, `torch.func`, and more.
4.  **Module 4: The `nn.Module`** - From basic building blocks to advanced patterns.
5.  **Module 5: Data Management** - `Dataset`, `DataLoader`, and custom data pipelines.
6.  **Module 6: Production & Deployment** - Serialization, TorchScript, and serving models.
7.  **Module 7: Performance & Profiling** - Tools and techniques for optimizing PyTorch code.

## Current Status

The project is in its initial phase. The first notebook, `01_Storage_vs_View.ipynb`, which covers tensor memory layout, is complete and serves as a quality benchmark for subsequent lessons.

## Key Files

- `README.md`: Provides the project's philosophy, goals, and setup instructions.
- `ROADMAP.md`: Contains the detailed curriculum and the full list of planned notebooks.
- `01_Storage_vs_View.ipynb`: The first completed lesson, exemplifying the course's content style.
- `pyproject.toml`: Defines project dependencies and the Python environment.

## Additional Coding Preferences

- Bleeding Edge Stack: Always prioritize the latest stable releases of all libraries. Avoid deprecated APIs (e.g., use weights=... instead of pretrained=True in Torchvision) and legacy patterns.
- CUDA 13.0 Target: All Deep Learning environment setups and installation commands must target CUDA 13.0 (cu130) compatibility.
- Modern PyTorch (2.x+): Leverage modern PyTorch paradigms, including torch.compile, Scaled Dot Product Attention (SDPA), and efficient memory management (contigous, pin_memory).
- uv Package Manager: Use uv exclusively for environment management (uv venv) and dependency installation (uv pip install), utilizing pyproject.toml for dependency tracking.
- Strict Typing: Enforce strict Python Type Hinting (e.g., def forward(self, x: torch.Tensor) -> torch.Tensor:) and use dataclasses for configuration objects to ensure code quality and clarity.
- Production-Ready Code: Favor modular design, Object-Oriented Programming (OOP) with nn.Module, and avoid "spaghetti code" even in notebooks.
