import torch 
import torch.nn as nn 
from torch.nn import functional as F 


#hyperparams
batch_size = 32         #from 16 shorter sequences with BPE so we can fit more 
block_size = 64         #from 32 subword carries more meaning than char
max_iters = 5000        
eval_interval = 100
learning_rate = 3e-4 
device = "cuda" if torch.cuda.is_available() else "cpu"  #will use vu servers so GPU for me 
eval_iters = 200
n_embd = 128 
n_head = 4
n_layer = 4 
dropout = 0.1 

torch.manual_seed(3456789)

#load text
with open('don-quixote.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#load the sentencepiece tokenizer we trained in the notebook
sp = spm.SentencePieceProcessor(model_file='quixote.model')
vocab_size = sp.get_piece_size()

encode = lambda s: sp.encode(s, out_type=int)
decode = lambda l: sp.decode(l)

#Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #90 train 10 val
train_data = data[:n]
val_data = data[n:]

#data loading 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    one head of self-attention
    """

    def __init__(self, head_size):
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_debuffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)
        q = self.query(x)

        #compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 #BTC @ BCT -> BTT
        wei = wei.masked_fill(self.tri[:T, :T] == 0, float("-inf")) #BTT
        wei = F.softmax(wei, dim=1) #BTT
        wei = self.dropout(wei)

        #perform the weighted aggregate of values 
        v = self.values(x)
        out = wei @ v #btt @ btx -> btc
        return out 
    


class FeedFoward(nn.Module):
    """
    a simple linear layer followed by a non linearlity
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    


class MultiHeadAttention(nn.Module):
    """
    multiple heads of self attention in parallel
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """
    transformer block comms followed by computation
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    one more class bro i promise broo
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def generate(self, idx, max_new_tokens):
        #idx is (B, T)  array of indeces in the current context

        for _ in range(max_new_tokens):
            
            #crop idx to the last block size tokens
            idx_cond = idx[:, -block_size]
            
            #get the predictions 
            logits, loss = self(idx_cond)

            #focus only on the last time stamp
            logits = logits[:, -1, :] #B,c

            #apply softmax to get proba
            probs = F.softmax(logits, dim=-1)

            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #B,1 

            #append sampled idx to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1 ) #B, T+1

        return idx 
    
model = GPTLanguageModel()
m = model.to(device)

print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M params")


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



for iter in range(max_iters):
    #every once in a while eval the loss on train and val sets 
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses["train"]:.4f}, val loss {losses["val"]:.4f}")


    #sample a batch of data 
    xb, yb = get_batch("train")

    
    #eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#gen model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))