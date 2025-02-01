# Genetic Algorithm with CUDA Acceleration

This project implements a **Genetic Algorithm (GA)** to optimize the **Rastrigin function**, a common benchmark in optimization problems. The algorithm supports **both CPU-based and GPU-accelerated** execution using **Numba CUDA**.

## 🚀 Features
- **Genetic Algorithm (GA)** for function optimization.
- **Parallelized GPU fitness evaluation** using CUDA.
- **CPU vs. GPU performance comparison**.
- **Configurable parameters** (population size, mutation rate, etc.).

## 📌 Requirements
- Python 3.7+
- NumPy
- Numba
- CUDA-compatible GPU & NVIDIA drivers (for GPU acceleration)

Install dependencies using:
```bash
pip install numpy numba
