# PDF2 — Vision-Language Model Efficiency (CLIP Case Study)

## 1. Introduction & Motivation
Vision-Language Models (VLMs) such as CLIP (Radford et al., 2021) combine visual and textual modalities into a shared embedding space.  
By aligning image and text representations through contrastive learning, they enable zero-shot recognition, retrieval, and cross-modal understanding.  

However, the visual backbone in most VLMs — typically a Vision Transformer (ViT) — suffers from **quadratic complexity O(n²)** with respect to the number of visual tokens.  
This high cost limits inference speed and memory efficiency, especially when processing high-resolution images or large-scale datasets.  

This section explores the computational bottlenecks in CLIP’s visual encoder and proposes lightweight mechanisms to improve **runtime and memory efficiency** while maintaining cross-modal performance.

---

## 2. Background & Related Work
### 2.1 CLIP Architecture Overview
CLIP consists of two main components:
- **Image Encoder:** a ViT-B/16 model that tokenizes the image into 16×16-patch embeddings and applies multi-head self-attention.  
- **Text Encoder:** a Transformer that encodes textual descriptions.  
Both encoders are trained jointly with a contrastive loss to align visual and textual features.

### 2.2 Transformer Complexity in Vision Encoders
The self-attention mechanism in ViTs computes pairwise relations among all tokens, leading to O(n²) runtime and memory cost.  
As the number of image patches increases with resolution, the computational burden grows rapidly, making ViT-based VLMs inefficient for deployment.

### 2.3 Efficiency-Oriented Approaches
Recent works attempt to reduce computation through:
- **Token Pruning:** dynamically discarding redundant patches (e.g., Dynamic ViT 2021, TOME 2023).  
- **Token Merging:** combining spatially similar tokens to shorten sequence length during inference.  
- **Lightweight CLIP variants:** EVA-CLIP (2023), MobileCLIP (2023) use distillation and weight sharing to achieve compactness.  
These studies show that visual tokens often contain redundant information that can be safely compressed.

### 2.4 Summary
Although these approaches improve computational efficiency, they often rely on complex dynamic controllers or retraining pipelines.  
A simpler, modular strategy for token reduction that preserves compatibility with pretrained CLIP remains under-explored.

---

## 3. Problem Statement & Motivation
CLIP’s ViT encoder processes roughly **50 × 50 = 2,500 visual tokens per image**, each undergoing multi-head attention at every layer.  
Empirically, many of these tokens represent low-importance regions (e.g., backgrounds or repetitive textures), yet they still participate in all attention computations.

This redundancy leads to:
1. **Quadratic runtime growth** with image resolution.  
2. **High GPU memory usage** due to attention maps.  
3. **Inefficient deployment** on edge or low-latency systems.

Our goal is to design an **Extension A: Vision Token Pruning** mechanism that removes or merges less informative tokens before attention computation,  
reducing latency and FLOPs without degrading alignment accuracy.

## 4. Proposed Extensions

To mitigate the computational bottleneck in CLIP’s visual encoder while preserving its alignment capability, we propose two complementary efficiency extensions:  
**Extension A – Vision Token Pruning (VTP)** and **Extension B – Cross-Modal Low-Rank Fusion (LoRA)**.  
Both are designed to be modular and easily integrated into pretrained CLIP models without retraining from scratch.

---

### 4.1 Extension A – Vision Token Pruning (VTP)

#### Motivation
The Vision Transformer (ViT) in CLIP processes all visual tokens equally, even those corresponding to redundant or low-information regions such as background or uniform textures.  
This results in quadratic attention cost O(n²).  
Inspired by recent token-reduction methods, we propose **Vision Token Pruning (VTP)**, which selectively removes less informative patches before each attention block.

#### Method
We assign each token an **importance score** based on its spatial activation magnitude, then retain only the top-p% tokens.  
Dropped tokens are replaced by a learnable “summary token” to preserve global context.  
This method requires no retraining of CLIP’s backbone and can be applied dynamically at inference time.

#### Pseudocode (Simplified)
```python
# x: patch embeddings [B, N, D]
# p: keep ratio (e.g., 0.7 means prune 30%)
scores = x.norm(dim=-1)                # token importance
keep_k = int(p * x.size(1))            # number of tokens to keep
top_idx = scores.topk(keep_k, dim=1).indices
x_pruned = x.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

# optional: add summary token for dropped regions
summary = x.mean(dim=1, keepdim=True)
x_pruned = torch.cat([summary, x_pruned], dim=1)

# feed pruned tokens into ViT encoder
y = vision_transformer(x_pruned)
````

#### Expected Benefits

* **Latency ↓ and Memory ↓** due to shorter token sequences.
* **Plug-and-Play:** works with pretrained CLIP checkpoints.
* **Minimal Accuracy Loss** if pruning ratio ≤ 30%.

This extension directly targets the quadratic bottleneck of CLIP’s visual encoder.

---

### 4.2 Extension B – Cross-Modal Low-Rank Fusion (LoRA)

#### Motivation

In CLIP-style VLMs, visual and textual embeddings are projected into a joint space via high-dimensional linear layers.
These cross-modal projections consume significant parameters and GPU memory.
To address this, we introduce a **low-rank re-parameterization** of the fusion layers.

#### Method

We decompose each fusion weight ( W \in \mathbb{R}^{d \times d} ) into two smaller matrices ( A \in \mathbb{R}^{d \times r} ) and ( B \in \mathbb{R}^{r \times d} ), where ( r \ll d ):
( W \approx A \times B ).
During fine-tuning, only ( A ) and ( B ) are updated while ( W ) remains frozen.
This effectively reduces trainable parameters from ( O(d^2) ) to ( O(2dr) ).

#### Pseudocode (Simplified)

```python
# z_v: visual embedding, z_t: text embedding
# replace high-d projection with low-rank adapters
A_v, B_v = nn.Linear(d, r), nn.Linear(r, d)
A_t, B_t = nn.Linear(d, r), nn.Linear(r, d)

z_v_fused = B_v(A_v(z_v))
z_t_fused = B_t(A_t(z_t))
similarity = cosine_similarity(z_v_fused, z_t_fused)
```

#### Expected Benefits

* **Parameter Efficiency ↑** (especially for fine-tuning).
* **Improved Memory Footprint ↓** during training.
* **Maintains Alignment Performance** due to low-rank adaptation.

Although Extension B is not empirically evaluated here, it represents a practical future direction for efficient multi-modal fine-tuning.

---

### 4.3 Summary of Extensions

| Extension                          | Target Component                  | Key Idea                                      | Expected Effect                            |
| ---------------------------------- | --------------------------------- | --------------------------------------------- | ------------------------------------------ |
| **A – Vision Token Pruning (VTP)** | CLIP Visual Encoder (ViT)         | Drop low-importance tokens before attention   | Latency ↓ , Memory ↓ , Minor Accuracy Loss |
| **B – Cross-Modal LoRA Fusion**    | CLIP Fusion Layer (Text ↔ Vision) | Low-rank factorization of projection matrices | Params ↓ , Fine-tuning Efficiency ↑        |

Together, these two strategies form a unified approach to **VLM efficiency optimization** —
Extension A addresses spatial redundancy in images, while Extension B reduces redundancy in cross-modal projection.

## 5. Experiment Setup

### 5.1 Objective
The purpose of this experiment is to evaluate the efficiency improvement introduced by **Extension A – Vision Token Pruning (VTP)** in CLIP’s visual encoder.  
We measure the trade-off between **latency**, **memory**, and **accuracy** under different pruning ratios, focusing on inference-time performance rather than fine-tuning accuracy.

---

### 5.2 Experimental Environment

| Parameter | Description | Values |
|------------|--------------|---------|
| **Model** | CLIP (ViT-B/16) from Hugging Face Transformers | `openai/clip-vit-base-patch16` |
| **Dataset** | CIFAR-10 subset (1 K samples) or synthetic images | Resolution = 128×128 |
| **Framework** | PyTorch 2.x + Hugging Face + CUDA 11.8 | Google Colab (T4 GPU / 16 GB RAM) |
| **Variants** | Baseline (no pruning) vs VTP p = 10%, 30%, 50% | Token retention ratios |
| **Metrics** | Latency (ms), Throughput (images/s), Peak Memory (MiB), Accuracy (%) | Averaged over 10 runs |

---

### 5.3 Implementation Details

The experiment applies token pruning **before the ViT encoder** during inference.  
Each image is patch-embedded, importance scores are computed by L2-norm across feature channels, and only the top-p% tokens are kept.  
The reduced sequence is then fed into the frozen CLIP visual backbone.  
This approach does not require retraining.

#### Pseudocode (Simplified)
```python
import torch, clip
from PIL import Image
from torchvision import transforms

# --- Load model ---
model, preprocess = clip.load("ViT-B/16", device="cuda")

# --- Prepare image ---
img = preprocess(Image.open("sample.jpg")).unsqueeze(0).cuda()
patch_embed = model.visual.conv1(img)           # [B, D, H, W]
tokens = patch_embed.flatten(2).transpose(1,2)  # [B, N, D]

# --- Token pruning ---
def prune_tokens(x, keep_ratio=0.7):
    scores = x.norm(dim=-1)
    k = int(keep_ratio * x.size(1))
    idx = scores.topk(k, dim=1).indices
    pruned = x.gather(1, idx.unsqueeze(-1).expand(-1,-1,x.size(-1)))
    summary = x.mean(dim=1, keepdim=True)
    return torch.cat([summary, pruned], dim=1)

tokens_pruned = prune_tokens(tokens, keep_ratio=0.7)
output = model.visual.transformer(tokens_pruned)
````

---

### 5.4 Measurement Protocol

To ensure consistent evaluation, we profile each variant using the following procedure:

1. **Warm-up:** 3 forward passes to stabilize GPU kernels.
2. **Measurement:** 10 forward runs averaged for latency and throughput.
3. **Memory tracking:** `torch.cuda.max_memory_allocated()` recorded per run.
4. **Accuracy (optional):** Compute top-1 accuracy on 1 K CIFAR-10 subset (zero-shot).

Example command structure in Colab:

```python
for p in [1.0, 0.9, 0.7, 0.5]:
    x_pruned = prune_tokens(tokens, keep_ratio=p)
    start = time.time(); _ = model.visual.transformer(x_pruned)
    torch.cuda.synchronize(); print(f"Ratio={p}, Time={(time.time()-start)*1000:.2f} ms")
```

---

### 5.5 Expected Outcome

We expect a **monotonic reduction in latency and memory** as pruning increases (lower p → fewer tokens),
with **minor accuracy degradation** when p ≥ 0.7.
The results will be summarized in Section 6 with comparison tables and visualizations similar to Part A.

---
