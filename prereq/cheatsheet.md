# GenAI Pre-work — Cheat Sheet
### Read this before the M00 session. Keep it open during class.

---

## Python You Will See Every Session

```python
# Variables
sentence = "The cat sat on the mat"
tokens   = sentence.lower().split()     # → ['the', 'cat', 'sat', ...]

# Lists
tokens[0]      # first item     → 'the'
tokens[-1]     # last item      → 'mat'
tokens[1:3]    # slice          → ['cat', 'sat']
len(tokens)    # count          → 6

# Loops
for token in tokens:
    print(token)              # runs once per token

# Functions
def tokenize(text):
    return text.lower().split()

result = tokenize("Hello World")    # → ['hello', 'world']

# Dictionaries
vocab = {'the': 0, 'cat': 1, 'sat': 2}
vocab['cat']                         # → 1  (look up word → ID)
for word, id in vocab.items():       # loop over key-value pairs
    print(f"{word} → {id}")

# f-strings
name = "Mohamed"
print(f"Hello {name}")               # → Hello Mohamed
```

---

## Math Concepts

### Vector
A list of numbers that represents something (a word, a sentence, an image).

```python
king  = [0.9, 0.1, 0.8]   # [royalty, gender, age]
queen = [0.9, 0.9, 0.8]   # similar to king!
pizza = [0.0, 0.5, 0.0]   # nothing in common
```

### Dot Product = Similarity
Multiply matching numbers, sum them up. Higher = more similar.

```python
import numpy as np
np.dot(king, queen)   # → 1.54  (similar!)
np.dot(king, pizza)   # → 0.05  (not similar)
```

### Cosine Similarity
Dot product normalised by vector sizes. Result is between -1 and 1 (for word vectors with positive values: 0 to 1).

```python
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite
```

### Softmax = Scores → Probabilities
Converts any numbers into probabilities that sum to 1.

```python
scores = np.array([8.5, 6.2, 2.1])     # raw model scores
probs  = np.exp(scores) / np.exp(scores).sum()
# → [0.91, 0.09, 0.00]  (sum = 1.0)
# The model picks the highest: index 0
winner = np.argmax(probs)               # → 0
```

---

## Neural Network Concepts

| Term | Plain English |
|---|---|
| **Neuron** | A math operation: weighted sum of inputs |
| **Weight** | A tunable number — like a dial |
| **Layer** | A group of neurons working in parallel |
| **Forward pass** | Data flows input → layers → output |
| **Classification** | Model picks one option from a list |
| **Loss** | How wrong the model is (0 = perfect) |
| **Gradient descent** | Nudge weights downhill → reduce loss |
| **Training** | Forward pass → compute loss → update weights → repeat |
| **Epoch** | One complete pass through all training data |

---

## The LLM Training Loop (Plain English)

```
For every word in all of the internet's text:
  1. Look at all previous words  (context)
  2. Predict the next word       (classification over 50,000 options)
  3. Compare to the actual word  (compute loss)
  4. Nudge all weights slightly  (gradient descent)

Repeat for months on thousands of GPUs.
Result: a model that is very good at predicting text.
```

---

## Key Terms for M00

| Term | What it means |
|---|---|
| **Token** | A piece of text (a word or word-fragment) |
| **Vocabulary** | The full list of tokens the model knows (~50,000) |
| **Embedding** | A token converted into a vector of numbers |
| **Attention** | A mechanism to measure how relevant each token is to others (dot products!) |
| **Transformer** | The neural network architecture used by all modern LLMs |
| **GPT** | A transformer trained to predict the next token, again and again |
| **Hallucination** | When the model predicts confidently but incorrectly |

---

## Before You Open M00

You should be able to answer YES to all of these:

- [ ] I can read a Python for loop and understand what it does
- [ ] I know what a dictionary is and how to look something up in it
- [ ] I understand that a vector is a list of numbers
- [ ] I know that dot product measures similarity between two vectors
- [ ] I understand that softmax converts scores to probabilities
- [ ] I know that a model learns by reducing its loss over many steps
- [ ] I understand that next-word prediction is a classification problem

If you answered NO to any of these, go back to the relevant notebook section.

---

*GenAI-2026 Batch | Pre-work complete → ready for M00*
