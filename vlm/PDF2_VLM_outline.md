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
