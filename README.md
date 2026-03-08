# Hierarchical-KV-Caching-in-Transformer-Architecture# 
Learned HiKV + ALiBi + LoRA Fine‑Tune: Constant‑Memory Long‑Context Inference

**Goal:** A practical, reproducible recipe to get **near Full‑KV quality** at **constant memory** and **flat latency**—even on a T4—by combining:

- **Learned Hierarchical KV (HiKV)**: recent tokens exact, older tokens compressed with a **learned attention pooler** (content‑aware summaries).
- **ALiBi**: attention linear bias for positions (compression‑friendly vs RoPE).
- **LoRA** fine‑tuning: adapt Q/K/V (and the pooler) to use compressed memory effectively.

This repo includes a Colab‑ready notebook, scripts, and a benchmarking harness to reproduce **throughput**, **peak VRAM**, and **decoded samples** across cache policies:
**FullCache**, **SlidingWindow**, and **Learned‑HiKV**.

---

## ✨ Why this matters

- **Full KV cache** scales linearly with context → slow and memory-heavy for long generations.
- **Sliding window** is constant memory but **forgets** older context.
- **Naive HiKV** (mean/EMA) keeps summaries but loses semantics.

**Learned HiKV** uses a small attention pooler to compress chunks **selectively** (preserving important tokens/themes), and **ALiBi** keeps position as a bias (not inside K/V). With a **light LoRA fine‑tune**, the model learns to **trust and use summaries**—often matching or exceeding sliding window quality at the same memory while keeping **constant latency**.

---

## 🔬 Method overview

**Hierarchical cache:**  
- **L0**: last `W0` tokens (exact)  
- **L1**: summaries of L0 chunks (size `W1`)  
- **L2**: summaries of L1 (size `W2`)  
- Total memory slots: `W0 + W1 + W2` (constant)

**Learned Pooler:**  
Instead of mean pooling,
```text
weights = softmax(q_pool · K_chunk)       # learned per-chunk importance
summary_k = weights @ K_chunk
summary_v = weights @ V_chunk
