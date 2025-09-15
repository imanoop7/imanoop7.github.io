---
title: "Transformers from Scratch: Implementing Attention Is All You Need"
date: 2025-09-15T12:00:00+00:00
tags: ["transformers", "pytorch", "nlp"]
author: "Anoop Maurya"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Beginner-to-intermediate guide to the Transformer with PyTorch code: attention, multi-head attention, positional encoding, and a minimal encoder."
canonicalURL: "https://imanoop7.github.io/posts/attention_transformer_blog/"
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

Transformers have transformed the field of sequence modeling by replacing recurrent nets with self-attention. I was curious about how this works under the hood, so I dove into the “Attention Is All You Need” paper and built the components from scratch in PyTorch. In this post I share my understanding and hands-on insights. I’ll cover the motivation behind Transformers, break down the core ideas, and show PyTorch snippets for the key modules: scaled dot-product attention, multi-head attention, positional encoding, and an encoder layer.

Along the way I’ll mention a few hiccups and “aha!” moments I had – hopefully making this journey relatable to anyone wrapping their head around this for the first time.

### Motivation

Before Transformers, sequence models like RNNs or LSTMs were standard for machine translation or language modeling. These models process data step-by-step, which is hard to parallelize and can struggle with long-range dependencies. The Transformer relies entirely on attention, allowing information from all time steps to be used at once.

- **Parallelism and efficiency**: Self-attention lets us process all tokens at once, so training is much faster on modern hardware.
- **Long-range context**: Every token can attend to any other token, capturing long-distance relationships without vanishing gradients.
- **Simplicity of components**: The building blocks (attention, feed-forward layers, etc.) are relatively simple operations—feasible to code and inspect directly.
- **Practical success**: Many cutting-edge models (BERT, GPT) are based on Transformers, so learning it deeply pays off.

### Core Concepts

The Transformer encoder is built from a few key ideas:

- **Scaled Dot-Product Attention**: Computes attention scores between queries and keys, scales them, and uses softmax to weight values.
- **Multi-Head Attention**: Runs several attention “heads” in parallel so the model can jointly attend to different representation subspaces.
- **Positional Encoding**: Injects information about token positions, since attention alone is order-agnostic.
- **Feed-Forward Network**: A two-layer MLP applied to each position separately and identically.
- **Add & Norm (Residual + LayerNorm)**: Residual connections and layer normalization for stable training and easy gradient flow.

Stacking several encoder layers (typically 6–12) gives you a Transformer encoder.

### Scaled Dot-Product Attention

Given queries Q, keys K, and values V, scaled dot-product attention computes a weighted sum of the values, where weights come from the similarity of queries with keys. The scores are divided by \(\sqrt{d_k}\) to keep values in a reasonable range.

High level steps for a single query \(\mathbf{q}\) and keys \(\mathbf{k}_j\):

1. Compute raw scores: \(\text{score}_j = \mathbf{q} \cdot \mathbf{k}_j^\top\)
2. Scale: \(\text{score}_j / \sqrt{d_k}\)
3. Optionally mask some scores to \(-\infty\)
4. Softmax to probabilities: \(\alpha_j\)
5. Weighted sum of values: \(\sum_j \alpha_j \mathbf{v}_j\)

PyTorch implementation:

```python
import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: shape (batch, seq_len, d_k)
    mask: optional mask tensor of shape (batch, seq_len_q, seq_len_k)
    Returns: (output, attention_weights)
    """
    d_k = Q.size(-1)
    # Compute scaled dot products
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply the mask (if provided) by setting scores to large negative where mask==0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # Softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    # Multiply by values
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

Here, `scores` has shape `(batch, seq_q, seq_k)` after the matmul. Multiplying by `V` then gives an output of shape `(batch, seq_q, d_k)`. For masks (e.g., padding or causal), use `masked_fill(-inf)` before softmax so those positions get effectively zero weight.

### Multi-Head Attention

One attention head gives one context representation. Multi-head attention runs several heads in parallel with separate linear projections, letting each head focus on different aspects (e.g., local vs. long-range patterns).

If `d_model` is the model dimension and we have `h` heads, each head uses `d_k = d_model / h`. There are learned projections `Wq, Wk, Wv` and an output projection `Wo`.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to project inputs to queries, keys, values
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Output linear layer
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1) Linear projections
        Q = self.Wq(Q)  # (batch, seq_len, d_model)
        K = self.Wk(K)
        V = self.Wv(V)

        # 2) Split into heads by reshaping
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now Q, K, V are (batch, heads, seq_len, d_k)

        # 3) Apply scaled dot-product attention on each head
        # Combine batch and heads dims for efficiency
        Q_flat = Q.reshape(-1, Q.size(2), self.d_k)  # (batch*heads, seq_q, d_k)
        K_flat = K.reshape(-1, K.size(2), self.d_k)
        V_flat = V.reshape(-1, V.size(2), self.d_k)

        if mask is not None:
            # Expand mask to match the number of heads
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.reshape(-1, mask.size(-2), mask.size(-1))

        context, attn_weights = scaled_dot_product_attention(Q_flat, K_flat, V_flat, mask)

        # 4) Concatenate heads
        context = context.view(batch_size, self.num_heads, -1, self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 5) Final linear layer
        output = self.Wo(context)  # (batch, seq_len, d_model)
        return output, attn_weights
```

The tensor reshaping is the trickiest part. After splitting into heads, I flatten `(batch, heads)` to reuse `scaled_dot_product_attention`, then reshape back and concatenate. Use `.contiguous()` before `.view` after a transpose.

### Positional Encoding

Pure attention has no notion of order. We add positional encodings to token embeddings. The original paper used fixed sinusoidal encodings:

\[\text{PE}(pos, 2i) = \sin\bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\bigr),\quad \text{PE}(pos, 2i+1) = \cos\bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\bigr)\]

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer so it’s saved with the model but not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

Shaping `pe` as `(1, max_len, d_model)` lets it broadcast-add to a batch of embeddings. Register it as a buffer so it moves with `model.to(device)` but isn’t updated by the optimizer.

### Putting It Together: Encoder Layer

Each encoder layer does:

1. Self-Attention + Add & Norm: `x <- LayerNorm(x + MHA(x, x, x))`
2. Feed-Forward + Add & Norm: `x <- LayerNorm(x + FFN(x))`

```python
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Feed-forward network (position-wise)
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x, mask=None):
        # 1) Multi-head self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual
        x = self.norm1(x)

        # 2) Feed-forward network
        ff_output = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        x = x + self.dropout(ff_output)  # Residual
        x = self.norm2(x)
        return x
```

If `x` is `(batch, seq_len, d_model)`, the module preserves shapes. Dropout is applied after attention and inside the feed-forward path.

### Reflections

- **Understand by implementing**: Writing code forces you to handle details (tensor shapes, masking). Quick dummy tests are invaluable.
- **Scaling matters**: Dividing by \(\sqrt{d_k}\) keeps softmax well-behaved; omitting it caused overly peaky distributions in my tests.
- **Reshaping is tricky but logical**: Splitting/combining heads is the most error-prone part; draw shapes while coding.
- **Positional encoding is neat**: The sine/cosine patterns give a smooth notion of distance; the model infers relative order.
- **Modularity**: Once `MultiHeadAttention` and `EncoderLayer` work, stacking them is straightforward and encourages experimentation.

Overall, the magic is that self-attention lets each position flexibly gather information from anywhere else, and multi-head attention enriches that process. If you’re learning this, don’t just read the paper—try coding the pieces. Small confusions often turn into big insights once resolved.



