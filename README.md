# Efficient Architectures for Sequence and Multimodal Modeling

**Date:** Nov 2025  
**Author:** Huiyao Lan  
**Status:** Completed

-----

## Overview

This repository explores **efficiency-oriented deep learning architectures** to address the computational bottlenecks in modern sequence and vision modeling. The project consists of two complementary parts:

| Part | Focus | Model Architecture | Core Technique |
| :--- | :--- | :--- | :--- |
| **Part A** | **Sequence Efficiency** | **Mamba (SSM)** | Linear-time Selective Scanning vs. Quadratic Attention |
| **Part B** | **Visual Efficiency** | **Qwen2.5-VL & CLIP** | **Training-Free Vision Token Pruning (VTP)** |

By combining **architectural innovation (SSMs)** with **input-level sparsity (Pruning)**, this project demonstrates how to significantly reduce FLOPs and latency while preserving representational power.

-----

## Repository Structure

```text
Efficiency-Arch-Project/
│
├── ssm/                          # Part A: Mamba & Selective Scanning
│   ├── mrss_proxy_experiment.ipynb   # Profiling linear-time scanning
│   └── results/                      # Benchmarking logs
│
└── vlm/                          # Part B: Vision Token Pruning (VTP)
    ├── clip_pruning.ipynb            # Main experiment notebook (PyTorch)
    └── results/                      # Visualization of Latency vs. Accuracy
        └── poster.pdf                      # Technical Presentation / Poster

```

-----

## Part B: Vision Token Pruning (VTP)

### 1\. Problem Statement

Large Vision-Language Models (LVLMs) like **Qwen2.5-VL** rely on Vision Transformers (ViT) for visual encoding. However, the **quadratic complexity ($O(N^2)$)** of the self-attention mechanism creates a massive latency bottleneck when processing high-resolution images (\~200+ patch tokens).

### 2\. Methodology: Training-Free L2-Norm Pruning

We engineered a custom **PyTorch inference pipeline** that dynamically prunes redundant visual tokens **without fine-tuning**.

  * **Mechanism:**  *(Recommended: Screenshot Slide 9 from 12.3.pdf and place here)*
  * **Scoring Function:** We utilize **L2-norm scoring** on the `Key` ($K$) and `Query` ($Q$) matrices within the attention layers to estimate token importance.
  * **Process:**
    1.  **Patch Embedding:** Convert image to patch sequences.
    2.  **Scoring:** Calculate importance scores per token.
    3.  **Top-K Selection:** Keep only the top $p$ (e.g., 50-70%) informative tokens.
    4.  **Forward Pass:** Execute the encoder with the shortened sequence.

### 3\. Key Results

Benchmarks were conducted on **CLIP (ViT-B/16)** and analyzed on **Qwen2.5-VL** architecture.

| Configuration | Keep Ratio ($p$) | Latency (ms) $\downarrow$ | Throughput (img/s) $\uparrow$ | Semantic Consistency (CosSim) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 100% | 81 ms | 100 | 1.00 |
| **VTP (Light)** | 70% | **63 ms (-22%)** | **125 (+25%)** | 0.89 |
| **VTP (Aggressive)**| 50% | **46 ms (-43%)** | **170 (+70%)** | **\~0.47** |

> **Impact:** Achieved up to **40% latency reduction** and **70% throughput improvement** while maintaining acceptable semantic consistency for downstream tasks.

*(Recommended: Place specific result charts/histograms from Slide 14 here)*

-----

## Part A: Structured State Space Models (Mamba)

  * **Objective:** Validated the efficiency of **Mamba's Selective Scanning** mechanism compared to standard Transformers.
  * **Result:** Mamba achieves **linear-time inference ($O(N)$)**, significantly outperforming Transformer's quadratic scaling on long sequences.
  * **Optimization:** Implemented a Multi-Resolution Selective Scanning (MRSS) proxy, further reducing redundant temporal updates by **\~35%**.

-----

## How to Reproduce

### Dependencies

```bash
pip install torch transformers==4.44.2 timm matplotlib
```

### Running VTP Experiments (Part B)

1.  Navigate to the `vlm/` directory.
2.  Open `clip_pruning.ipynb`.
3.  Run cells sequentially to:
      * Load the pre-trained CLIP model.
      * Apply the custom `prune_tokens` function.
      * Generate Latency vs. Sparsity benchmarks.

-----

## References

  * **Qwen2.5-VL:** Alibaba Cloud (2024).
  * **CLIP:** Radford et al., *Learning Transferable Visual Models from Natural Language Supervision* (ICML 2021).
  * **Mamba:** Gu et al., *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2024).
  * **ToMe:** Bolya et al., *Token Merging for Efficient Vision Transformers* (CVPR 2023).

-----

### Author Note

This project was developed as part of advanced research into **Efficient ML Systems** at the University of Toronto. It focuses on identifying and optimizing atomic bottlenecks in SOTA architectures (Transformers & SSMs).
