#@title ⚡ Streaming benchmark (bf16 AMP), safe top-p filtering
def top_p_filtering(logits, top_p=0.9, temperature=1.0):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits / max(1e-6, temperature), dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumprobs > top_p
    remove[..., 1:] = remove[..., :-1].clone()  # critical fix: avoid overlapping write
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float('inf'))
    filtered = torch.full_like(logits, -float('inf'))
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered

@torch.no_grad()
def stream_benchmark_t4(model, cache_policy, start_token, steps=4096, temperature=1.0, top_p=0.9, dtype='bf16'):
    model.eval()
    # assign per-layer independent cache states
    model.set_cache_policy(cache_policy)
    model.reset_cache(B=start_token.shape[0], device=device)
    reset_peak()

    idx = start_token.clone().to(device)  # [B,1]
    toks = []
    amp_dtype = torch.bfloat16 if dtype=='bf16' else torch.float16

    # warm-up
    cuda_sync()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=amp_dtype):
        for _ in range(32):
            logits, _ = model(idx, use_cache=True)
            logits_step = logits[:, -1, :]
            if top_p < 1.0:
                filtered = top_p_filtering(logits_step, top_p=top_p, temperature=temperature)
                probs = F.softmax(filtered, dim=-1)
            else:
                probs = F.softmax(logits_step / temperature, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)

    cuda_sync()
    t0 = time.perf_counter()

    # timed run
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=amp_dtype):
        for _ in range(steps):
            logits, _ = model(idx, use_cache=True)
            logits_step = logits[:, -1, :]
            if top_p < 1.0:
                filtered = top_p_filtering(logits_step, top_p=top_p, temperature=temperature)
                probs = F.softmax(filtered, dim=-1)
            else:
                probs = F.softmax(logits_step / temperature, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            toks.append(idx.item())

    cuda_sync()
    dt = time.perf_counter() - t0
    return {
        "tokens": toks,
        "tokens_sec": steps / dt,
        "latency_ms": 1000.0 * dt / steps,
        "peak_mem_mb": peak_mem_mb()
    }