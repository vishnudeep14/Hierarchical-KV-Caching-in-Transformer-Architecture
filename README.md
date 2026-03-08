# Hierarchical-KV-Caching-in-Transformer-Architecture 
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
```

The pooler is a tiny attention head (negligible overhead), shared across layers.
ALiBi:
Adds a linear positional bias to attention scores (no position embedded inside K/V), making compression far more robust.
LoRA FT (Q/K/V + pooler):

Freeze base model
Train only LoRA A/B for Q/K/V and the pooler (few minutes on T4)
Adaptation aligns the model to compressed memory


What you can expect

Throughput: flat vs generated length for Learned‑HiKV and Sliding; FullCache degrades.
Peak VRAM: flat for Learned‑HiKV and Sliding; FullCache grows with length.
Quality: Learned‑HiKV ≥ Sliding at equal memory budgets (after LoRA FT); often close to FullCache for many tasks on Tiny Shakespeare.
----

On small models trained from scratch without adaptation, Sliding can appear better than HiKV. Learned compression + ALiBi + LoRA FT closes that gap.
---


**🧪 Quick start (Colab / T4)**
The fastest way is to run the Colab notebook in notebooks/learned_hikv_alibi_lora_colab.ipynb, which contains everything:

data loading (Tiny Shakespeare)
model (ALiBi + LoRA in Q/K/V)
Learned HiKV cache policy
LoRA fine‑tune loop (streaming teacher forcing)
T4 AMP (bf16) streaming benchmark
plots + decoded samples
CSV export


---

**🧱 Repo contents**

src/model.py: GPT‑Mini with ALiBi and LoRA Q/K/V; merged‑head streaming cache path.
src/cache_policies.py: FullCache, SlidingWindow, HiKV_Learned (with learned pooler).
src/pooler.py: LearnedPooler (q_pool attention).
src/alibi.py: ALiBi slopes & bias helpers.
src/lora.py: LoRA wrappers and utilities.
src/train_ft.py: Streaming teacher‑forcing fine‑tune loop (LoRA + pooler only).
src/bench.py: Streaming benchmark harness; CSV + plots.
src/data.py: Tiny Shakespeare data loader.
src/utils.py: reproducibility utils (seed, AMP, CUDA sync).

The Colab notebook contains a single‑file version of these components.

---

**🧪 Long-range recall**
A minimal Needle‑in‑a‑Haystack (NIAH) probe is provided in src/bench.py:

teacher‑forces a long random context
checks next‑token match

At equal budgets:
FullCache ≥ Learned‑HiKV ≳ EMA ≥ Sliding


For stronger tests, cue the needle (e.g., delimiters + “repeat the token after ###”) or use structured copy tasks.


----

**🧠 Design notes**

Merged‑head cache: For simplicity and speed, we cache merged heads [B,1,C] during streaming. ALiBi bias uses a mean slope approximation. For maximum fidelity, per‑head caches can be implemented (more code/VRAM).
Why ALiBi, not RoPE? RoPE encodes position inside K/V; compression breaks this. ALiBi pushes position to the score bias, robust to K/V compression.
Why LoRA FT? You change the K/V distribution with learned summaries; a light LoRA update teaches the model how to route and trust summaries.
---

🧪 Results
Throughput: HiKV_Learned ~108 tok/s (flat), Sliding_L=512 ~88.2 tok/s (flat), FullCache ~62 tok/s (degrades with length)
VRAM: HiKV_Learned / Sliding flat around ~68‑73 MB on T4; FullCache grows with length
Samples: Learned‑HiKV maintains global topic better than Sliding at the same memory; fewer degeneracies

----

