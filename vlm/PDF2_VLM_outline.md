# PDF2 — Vision-Language Model Efficiency (CLIP Case Study)

## 1. Introduction & Motivation
- Vision-Language Models (VLMs) combine image encoders and text encoders for cross-modal alignment.  
- Despite strong performance, most VLMs inherit the quadratic cost of Transformers in the visual backbone (ViT).  
- This section explores efficiency bottlenecks in CLIP and proposes methods to reduce redundant visual tokens while preserving accuracy.

## 2. Background & Related Work
- CLIP architecture overview (image encoder + text encoder + contrastive objective).  
- Transformer quadratic complexity in Vision Encoders.  
- Token Pruning and Merging approaches (TOME, Dynamic ViT, EVA-CLIP, MobileCLIP).  
- Summary of efficiency trade-offs in recent VLM optimizations.

## 3. Problem Statement & Motivation
- CLIP ViT encoders process ~50×50 = 2500 tokens per image.  
- Many tokens represent redundant spatial information, causing unnecessary attention computation.  
- Goal: reduce visual token count without hurting alignment accuracy.

## 4. Proposed Extensions
### 4.1 Extension A — Vision Token Pruning
- Apply token importance scoring (SALIENCY / variance / attention entropy).  
- Discard or merge least-informative tokens before ViT layers.  
- Evaluate speed vs accuracy trade-off.

### 4.2 Extension B — Cross-Modal Low-Rank Fusion
- Replace full attention in text–vision fusion with low-rank projection (LoRA).  
- Compress interaction parameters while maintaining semantic alignment.

## 5. Experiment Setup (CLIP Proxy)
- Use Hugging Face `openai/clip-vit-base-patch16`.  
- Measure forward time and GPU memory for baseline vs pruned CLIP.  
- Dataset: CIFAR-10 subset or synthetic images (64×64).  
- Metrics: Latency, Throughput, Memory.

## 6. Results & Analysis
- Table and bar charts comparing baseline vs Extension A.  
- Observe ~30–40% latency reduction with minimal accuracy drop.  
- Discuss scaling behavior vs sequence length.

## 7. Conclusion & Future Work
- Token pruning effectively reduces VLM inference cost.  
- Next steps: dynamic token budget per image, integration with LoRA for cross-modal compression.  
- Connection to Part A: both address efficiency of attention-free or token-reduced models.

