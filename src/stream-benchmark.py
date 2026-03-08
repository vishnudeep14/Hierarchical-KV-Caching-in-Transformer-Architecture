#@title 🧪 Scaling: FullCache vs Sliding vs Learned-HiKV; plots + samples
import pandas as pd
import matplotlib.pyplot as plt

# Build policies (note: we want per-layer independent states)
full = FullCache()
slide84 = SlidingWindow(L=84)
slide512 = SlidingWindow(L=512)
# Learned HiKV shares pooler across layers (via new_empty_like makes layer-local state)
hikv_learned = HiKV_Learned(W0=64, W1=16, W2=4, embed_dim=model.embed_size, pooler=model.hikv_pooler)

policies = {
    "FullCache": full,
    "Sliding_L=84": slide84,
    "Sliding_L=512": slide512,
    "HiKV_Learned_64_16_4": hikv_learned,
}

lengths = [1024, 2048]
TEMPERATURE = 1.0
TOP_P = 0.9
DTYPE = 'bf16'
start_char = '\n' if '\n' in stoi else random.choice(vocab)
start_tok = torch.tensor([[stoi[start_char]]], device=device)

rows = []
samples_once = {}

for L in lengths:
    for name, pol in policies.items():
        res = stream_benchmark_t4(model, pol, start_tok, steps=L, temperature=TEMPERATURE, top_p=TOP_P, dtype=DTYPE)
        rows.append({"len": L, "policy": name,
                     "tokens_sec": res["tokens_sec"],
                     "latency_ms": res["latency_ms"],
                     "peak_mem_mb": res["peak_mem_mb"]})
        if name not in samples_once:
            samples_once[name] = res
        print(f"{L:6d} | {name:18s} | {res['tokens_sec']:.1f} tok/s | {res['latency_ms']:.2f} ms | {res['peak_mem_mb']:.1f} MB")

df = pd.DataFrame(rows)
df.to_csv("t4_learned_hikv_alibi_lora_bench.csv", index=False)
print("\nSaved: t4_learned_hikv_alibi_lora_bench.csv")

# plots
plt.figure(figsize=(8,4))
for name in policies.keys():
    sub = df[df.policy==name]
    plt.plot(sub["len"], sub["tokens_sec"], marker='o', label=name)
plt.xscale('log', base=2)
plt.xlabel("Generated length (tokens)"); plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs. generated length (T4, AMP bf16)")
plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3); plt.show()

plt.figure(figsize=(8,4))
for name in policies.keys():
    sub = df[df.policy==name]
    plt.plot(sub["len"], sub["peak_mem_mb"], marker='o', label=name)
plt.xscale('log', base=2)
plt.xlabel("Generated length (tokens)"); plt.ylabel("Peak GPU memory (MB)")
plt.title("Peak VRAM vs. generated length (T4)")
plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3); plt.show()

# decoded sample previews
def decode_preview(ids, n_chars=300):
    s = decode(ids[:n_chars]); return s.replace("\r","␍").replace("\n","␤")

print("\n=== 300-char decoded samples (same temperature/top_p) ===")
for name, res in samples_once.items():
    print(f"\n--- {name} ---")
    print(decode_preview(res["tokens"], 300))