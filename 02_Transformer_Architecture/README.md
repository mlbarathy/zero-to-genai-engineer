# Session 02 — Transformer Architecture

> **The big question:** How does the same word mean different things in different sentences — and how does a model know the difference?

In Session 01, every embedding method gave each word one fixed vector. `"bank"` (river) and `"bank"` (finance) got the same representation. The Transformer solves this with **self-attention** — a mechanism that builds a new, context-aware vector for each word by looking at every other word in the sentence simultaneously.

This session builds an Encoder-Decoder Transformer from scratch in PyTorch, following the original *Attention is All You Need* paper. By the end, you will have trained a working English → Italian translation model.

---

## What's MISSING after this session?

The Transformer solves context-dependence — but training one from scratch takes days and enormous compute.

In practice, nobody trains a Transformer from scratch. They take a **pre-trained model** (GPT, BERT, Gemini) and interact with it via an API — passing carefully crafted text called a **prompt** to control what it produces.

That gap — **how to use a pre-trained LLM efficiently** — is what the next sessions cover: tokenization, context windows, embeddings, and API calls.

---

## Notebooks

Run in **Google Colab** (recommended — requires GPU for the training section).

| # | Notebook | What You Build | Time |
|---|---|---|---|
| 01 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nursnaaz/zero-to-genai-engineer/blob/main/02_Transformer_Architecture/notebooks/01_transformer_from_scratch.ipynb) [transformer_from_scratch.ipynb](notebooks/01_transformer_from_scratch.ipynb) | Full Encoder-Decoder Transformer in PyTorch — embeddings, positional encoding, multi-head attention, encoder/decoder stacks, training loop on English → Italian | 90 min |

**Requires:** GPU runtime in Colab (`Runtime → Change runtime type → T4 GPU`). The building-blocks section runs on CPU; training requires GPU.

---

## Slides

| File | Contents |
|---|---|
| [Transformers.pptx.pdf](slides/Transformers.pptx.pdf) | Full Transformer architecture walkthrough — attention, encoder-decoder, positional encoding |

---

## Papers

| File | What It Is |
|---|---|
| [Attention_Is_All_You_Need.pdf](papers/Attention_Is_All_You_Need.pdf) | The original 2017 paper by Vaswani et al. — read alongside the notebook |

---

## Assets

| File | What It Is |
|---|---|
| [self_attention_animation.gif](assets/self_attention_animation.gif) | Animated visualisation of self-attention across tokens |
| [SelfAttentionFull.mp4](assets/SelfAttentionFull.mp4) | Full self-attention walkthrough video |

---

## What the Notebook Covers

The notebook follows the architecture diagram from the paper step by step:

| Component | What You Build |
|---|---|
| `InputEmbeddings` | Token → dense vector (d_model = 512) |
| `PositionalEncoding` | Sine/cosine positional signal added to embeddings |
| `LayerNormalization` | Stabilises training — why epsilon matters |
| `ResidualConnection` | Skip connections to prevent vanishing gradients |
| `FeedForwardBlock` | Two-layer MLP after each attention block |
| `MultiHeadAttention` | Q / K / V projections, scaled dot-product, 8 heads |
| `EncoderBlock` + `EncoderStack` | 6 stacked encoder layers |
| `DecoderBlock` + `DecoderStack` | 6 stacked decoder layers with cross-attention |
| `LinearProjectionLayer` | Projects decoder output to vocabulary logits |
| `Transformer` + `build_transformer()` | Full assembled model |
| Training loop | English → Italian translation on `opus_books` dataset |

---

## Setup

```bash
# Install dependencies (or use Colab — no local setup needed)
pip install torch transformers tokenizers datasets tqdm

# Open the notebook
jupyter notebook notebooks/01_transformer_from_scratch.ipynb
```

For Colab: click the **"Open in Colab"** badge above and select `Runtime → Change runtime type → T4 GPU`.
