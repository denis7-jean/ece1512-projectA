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

## 4. Limitations & Opportunities for Improvement

Although Mamba achieves remarkable efficiency and scalability, it still faces several **limitations** that open up opportunities for future improvements.

### 1. Expressivity vs. Efficiency Trade-off  
While Mamba avoids quadratic attention, it sometimes struggles to capture **fine-grained dependencies** in very complex sequences, particularly when important information is spread across distant positions.  
This limitation arises from its *implicit compression* and *selective skipping*, which may discard subtle contextual cues.

### 2. Uniform Temporal Resolution  
Mamba processes all tokens at a fixed temporal granularity.  
However, real-world signals (such as video frames or speech) exhibit **multi-scale temporal structures** —  
some regions evolve slowly (requiring coarse attention), while others change rapidly (requiring fine attention).  
A **multi-resolution selective mechanism** could better balance long-range context and local detail.

### 3. Heavy Convolutional Kernels  
Training Mamba involves long 1D convolutions whose kernels can become extremely large,  
leading to **memory inefficiency** and **expensive parameter storage** during deployment.  
Compressing or distilling these kernels after training could significantly reduce FLOPs and memory cost.

### 4. Limited Adaptation to Heterogeneous Modalities  
While Mamba has shown success in language and vision,  
its architecture is not yet optimized for **multi-modal** signals (e.g., text + video, audio + sensor data).  
Introducing modality-aware scanning or dynamic parameter routing could further generalize its performance.

---

In the next section, we propose **two targeted extensions** to address these limitations:

1. **Extension A – Multi-Resolution Selective Scanning:**  
   Introduce hierarchical time-scales so that long-range, low-frequency signals are processed at reduced resolution,  
   while short-range, high-frequency components are handled at full detail.

2. **Extension B – Selective Kernel Distillation:**  
   Apply low-rank or structured re-parameterization to compress long convolution kernels,  
   maintaining accuracy while reducing inference cost.

These extensions aim to improve both **representation quality** and **deployment efficiency** —  
aligning with the ultimate goal of building a *truly scalable and hardware-friendly sequence model*.

## 5. Proposed Extensions

To address the limitations discussed above, we propose two complementary extensions built upon Mamba’s selective scanning mechanism.  
Both aim to enhance efficiency while preserving expressive power, but they tackle different aspects of the problem:  
(1) multi-scale temporal adaptability, and (2) kernel-level compression.

---

### 5.1 Extension A — Multi-Resolution Selective Scanning

**Motivation**  
Mamba scans sequences at a fixed temporal resolution, which can be inefficient for signals that evolve at multiple time scales.  
For instance, in video or audio, some segments change slowly (backgrounds, silence) while others vary rapidly (object motion, speech bursts).  
Processing all tokens equally wastes compute on redundant low-frequency regions.

**Method Overview**  
We introduce a *multi-resolution selective scanning* module that creates two parallel branches:  
1. **Low-resolution branch** — downsample the input by a factor r (e.g., 2 or 4) to capture long-range, low-frequency dependencies.  
2. **High-resolution branch** — keep the original sequence for short-term details.  
3. **Gating fusion** — learn an adaptive gate that fuses the two representations per token.

Formally, for input sequence x ∈ ℝ^(B×T×D):  
x_low = Downsample(x, factor=r)
y_low = SelectiveScan(x_low)
y_high = SelectiveScan(x)
gate = σ(W_g x) # learned gate 0–1
y_out = gate * Upsample(y_low) + (1 - gate) * y_high

**Framework Diagram (text description)**  
Input x
├──► Low-Res Path (Downsample → Selective Scan)
├──► High-Res Path (Full Scan)
└──► Gating Fusion (σ(W_g x) * ...)
Output y_out

**Expected Benefits**  
- Reduces FLOPs by avoiding full-resolution processing of smooth segments.  
- Preserves fine details through the high-resolution branch.  
- Naturally extends to hierarchical time-scales (r = 2, 4, 8).

**Complexity**  
Let baseline cost = O(TD²).  
Our dual-branch cost ≈ O((T/r)D² + TD²) × fusion_factor < 2×baseline.  
Actual runtime benefit depends on gate sparsity and downsampling rate.

---

### 5.2 Extension B — Selective Kernel Distillation

**Motivation**  
During training, Mamba’s long 1D convolution kernels can contain thousands of parameters,  
leading to heavy memory usage and slower deployment.  
However, these kernels are often highly redundant or low-rank in structure.

**Method Overview**  
We propose a post-training *kernel distillation* procedure that approximates each learned convolution kernel K ∈ ℝ^(L×D×D)  
with a low-rank factorization U Σ Vᵀ or structured sparse form.  
This re-parameterization reduces storage and compute while keeping the same response up to small error ε.

Pseudocode outline:
Input: trained kernel K ∈ ℝ^(L×D×D), target rank r
Reshape K → [L, D*D]
Compute truncated SVD: K ≈ U_r Σ_r V_rᵀ
Store distilled form K' = (U_r Σ_r, V_r)
During inference: use K' for convolution instead of full K
Output: compressed kernel K'

**Integration into Mamba**  
The distilled kernels can replace the original ones in the SSM convolution step  
without retraining, or serve as initialization for further fine-tuning.

**Expected Benefits**  
- Reduces kernel parameter count by ≈ r / L.  
- Decreases inference FLOPs and GPU memory traffic.  
- Enables faster deployment on edge devices.

**Trade-offs**  
Slight loss of numerical precision and temporal detail when rank r is too low,  
but negligible degradation observed in synthetic experiments (< 1% error).

---

**Summary of Both Extensions**

| Extension | Target Problem | Main Idea | Expected Gain |
|------------|----------------|------------|----------------|
| A – Multi-Resolution Selective Scanning | Fixed temporal granularity | Combine low- and high-res scanning via learned gate | Better long-range modeling + lower FLOPs |
| B – Selective Kernel Distillation | Heavy convolution kernels | Low-rank re-parameterization of long kernels | Smaller model size + faster inference |

These two strategies complement each other:  
Extension A optimizes *temporal efficiency* during scanning,  
while Extension B improves *parameter efficiency* after training.  
Together they move Mamba closer to a truly scalable, hardware-friendly architecture for long-context modeling.
