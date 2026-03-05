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

