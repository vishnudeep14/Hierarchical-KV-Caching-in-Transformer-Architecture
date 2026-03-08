#@title 📚 Load Tiny Shakespeare (char-level)
import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(URL, timeout=10).text

vocab = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(vocab)}
itos = {i:c for i,c in enumerate(vocab)}
vocab_size = len(vocab)

encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train, val = data[:split], data[split:]

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)