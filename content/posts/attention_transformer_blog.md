---
title: "Attention Is All You Need (Transformer Explained)"
date: 2025-09-15T12:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Anoop Maurya"
author: ["Me"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "This blog gives a beginner-friendly introduction to the Attention Is All You Need paper."
canonicalURL: "https://anoopmaurya.github.io/content/posts/attention_transformer_blog.md"
disableHLJS: false # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
---

# Attention Is All You Need (Transformer Explained)

The **Transformer** is a deep learning model that changed the way we deal with sequences like text.  
Unlike RNNs or LSTMs, it does **not** use recurrence. Instead it uses only **attention**.  
This makes it faster, more parallelizable, and better at handling long-range dependancies.  

---

## Why not RNNs or LSTMs?
- They process one token at a time → slow.  
- They forget things that happend far in the past.  
- Hard to train on long sequnces.  

**Transformer**: all tokens attend to each other at once.  
**RNN/LSTM**: step by step, so long distance info can get lost.  

---

## Scaled Dot-Product Attention

Attention is the core idea.  
Each word makes a **query** asking "which other words are important?".  
It compares to all **keys** and gathers info from their **values**.  

Steps:
1. Compute scores = Q @ K^T  
2. Scale by sqrt(d_k)  
3. Mask unwanted positions (optional)  
4. Softmax to get weights  
5. Weighted sum with V  

```python
import torch
import math

def attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## Multi-Head Attention

Instead of one attention, we do it in parallel with multiple heads.  
Each head can focus on different relations (like grammar or meaning).  

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        bsz = Q.size(0)
        Q = self.q_lin(Q).view(bsz, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.k_lin(K).view(bsz, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.v_lin(V).view(bsz, -1, self.num_heads, self.d_k).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(bsz, -1, self.num_heads*self.d_k)
        return self.out_lin(out)
```

---

## Positional Encoding

Since there is no recurrence, we need to tell the model the **order** of words.  
We add a sinusoidal encoding to the embeddings.  

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import torch
        import math
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## Encoder Layer (simplified)

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
```
