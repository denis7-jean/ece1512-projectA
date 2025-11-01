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
