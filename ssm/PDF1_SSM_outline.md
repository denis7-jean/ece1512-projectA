# PDF1 — SSM (Mamba) Outline

## 1. Introduction & Motivation  

**English:**  
Transformers have become the de-facto architecture for sequence modeling across language, vision, and multimodal tasks.  
However, their *quadratic* computational complexity with respect to sequence length (O(n²)) makes them inefficient for long-context applications.  
State Space Models (SSMs), such as S4 (Gu et al., 2021), propose an alternative approach:  
they model long-term dependencies through a **linear-time recurrent formulation** derived from control theory,  
achieving *efficient inference* while maintaining expressive power.  

Recent models like **Mamba (2024)** further extend this idea by combining the convolutional nature of SSMs with *selective scanning* and *input-dependent recurrence*.  
This design allows Mamba to adaptively focus on important parts of the sequence, achieving **sub-quadratic efficiency** without relying on attention mechanisms.  

## 2. Background & Related Work  
- **S4 (Structured State Space Sequence model, 2021):** classical linear time-invariant (LTI) ODE formulation for deep learning.  
- **S4ND / Diagonal SSM:** multi-dimensional and memory-efficient variants.  
- **Vision Mamba (2024):** extends SSMs to visual data using patchified inputs.  
- **Goal of Mamba:** balance efficiency, stability, and expressive power while avoiding full attention computation.  

## 3. Mamba Technical Summary

Mamba builds upon the S4 family of State Space Models but introduces **selective scanning** — an input-dependent mechanism that dynamically decides which parts of the sequence should be updated or skipped.  

### Key Design Components
1. **Input-Dependent Parameters**  
   Traditional SSMs use fixed transition matrices (A, B, C) across the entire sequence.  
   Mamba makes them **time-varying** and **input-conditioned**, allowing the model to adapt to changing signal dynamics.

2. **Selective Scanning (Focus)**  
   Instead of processing every token equally, Mamba learns *gates* that control which tokens are worth updating.  
   This acts like a compression step — improving efficiency by skipping redundant information.

3. **Hardware-Aware Parallelism**  
   Mamba reformulates its recurrence equations so that they can be parallelized efficiently on GPUs,  
   leading to **throughput comparable to transformers** while maintaining linear-time complexity during inference.

4. **Convolutional View**  
   During training, Mamba treats its recurrence as a *long 1D convolution*, enabling the use of FFT-based optimization  
   (similar to S4). This bridges RNN-style recurrence and convolutional training.

### Summary
Mamba can be interpreted as:
- A **recurrent model** for inference (efficient and scalable), and  
- A **convolutional model** for training (parallel and stable).  

This dual interpretation allows it to retain the best of both worlds — *speed* and *context length* — positioning it as a promising alternative to attention-based architectures for very long sequences.
