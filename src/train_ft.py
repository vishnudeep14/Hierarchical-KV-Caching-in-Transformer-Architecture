#@title 🏗️ Build model; (optional) load a weights checkpoint
EMBED = 384; LAYERS = 6; HEADS = 6; BPTT = 256
model = GPTMini(vocab_size, embed_size=EMBED, n_layers=LAYERS, n_heads=HEADS, block_size=BPTT, dropout=0.1,
                lora_r=8, lora_alpha=16).to(device)

# (Optional) Load a pre-trained checkpoint of GPTMini (trained without ALiBi).
# If you have one:
# state = torch.load("/content/gptmini_tinyshakespeare.pth", map_location='cpu')
# model.load_state_dict(state, strict=False)
# model.eval()

# quick val loss with full attention (ALiBi active)
@torch.no_grad()
def eval_ppl(model, dataset, block_size=256, batch_size=256, iters=50):
    model.eval()
    losses=[]
    for _ in range(iters):
        xb, yb = get_batch(dataset, batch_size, block_size)
        _, loss = model(xb, yb, use_cache=False)
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    bpc = mean_loss / math.log(2)
    ppl = math.exp(mean_loss)
    print(f"Val Loss {mean_loss:.3f} | PPL {ppl:.2f} | BPC {bpc:.3f}")
    return dict(loss=mean_loss, ppl=ppl, bpc=bpc)

_ = eval_ppl(model, val, BPTT, 256, iters=20)