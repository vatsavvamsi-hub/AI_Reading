# Deep Learning Numerical Example

A comprehensive walkthrough of training a simple neural network with backpropagation, including detailed matrix mathematics.

---

## Table of Contents
1. [Problem Setup](#problem-setup)
2. [Forward Pass](#forward-pass)
3. [Loss Computation](#loss-computation)
4. [Backward Pass (Backpropagation)](#backward-pass-backpropagation)
5. [Weight Updates](#weight-updates)
6. [Why We Use Matrices](#why-we-use-matrices)
7. [Key Takeaways](#key-takeaways)

---

## Problem Setup

**Task:** Binary classification (e.g., predicting if an email is spam)

**Network Architecture:**
- **Input layer:** 2 neurons (features: x₁, x₂)
- **Hidden layer:** 2 neurons with **ReLU** activation
- **Output layer:** 1 neuron with **Sigmoid** activation
- **Loss function:** Binary Cross-Entropy

```
Input (2)  →  Hidden (2, ReLU)  →  Output (1, Sigmoid)  →  Loss
```

### Initial Values

**Input and Target:**
```
x = [0.5, 0.3]    (input features)
y = 1             (true label: positive class)
```

**Weights and Biases:**

Layer 1 (Input → Hidden):
```
W₁ = | 0.1   0.2 |     b₁ = | 0.1 |
     | 0.3   0.4 |          | 0.2 |
```

Layer 2 (Hidden → Output):
```
W₂ = [0.5, 0.6]        b₂ = [0.1]
```

**Learning rate:** η = 0.1

---

## Forward Pass

### Step 1.1: Input to Hidden Layer

**Linear combination (z₁):**
```
z₁ = W₁ · x + b₁
```

Calculate each element:
```
z₁[0] = (0.1 × 0.5) + (0.2 × 0.3) + 0.1
      = 0.05 + 0.06 + 0.1 = 0.21

z₁[1] = (0.3 × 0.5) + (0.4 × 0.3) + 0.2
      = 0.15 + 0.12 + 0.2 = 0.47
```

**Result:** `z₁ = [0.21, 0.47]`

### Step 1.2: Apply ReLU Activation

**ReLU function:**
```
ReLU(z) = max(0, z)
```

**Hidden layer output (h):**
```
h[0] = max(0, 0.21) = 0.21
h[1] = max(0, 0.47) = 0.47
```

**Result:** `h = [0.21, 0.47]`

### Step 1.3: Hidden to Output Layer

**Linear combination (z₂):**
```
z₂ = W₂ · h + b₂
   = (0.5 × 0.21) + (0.6 × 0.47) + 0.1
   = 0.105 + 0.282 + 0.1
   = 0.487
```

### Step 1.4: Apply Sigmoid Activation

**Sigmoid function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Predicted output (ŷ):**
```
ŷ = σ(0.487) = 1 / (1 + e^(-0.487))
  = 1 / (1 + 0.6145)
  = 1 / 1.6145
  = 0.6194
```

**Prediction:** ŷ = 0.6194 (62% confidence for class 1)

---

## Loss Computation

**Binary Cross-Entropy Loss:**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Since y = 1:
```
L = -log(ŷ) = -log(0.6194) = 0.4794
```

---

## Backward Pass (Backpropagation)

We compute gradients using the **chain rule**, propagating from output to input.

### Step 3.1: Output Layer Gradients

**Gradient of Loss w.r.t. ŷ:**
```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
      = -1/0.6194
      = -1.6145
```

**Gradient of Sigmoid:**
```
∂ŷ/∂z₂ = ŷ(1 - ŷ) = 0.6194 × 0.3806 = 0.2357
```

**Combined gradient (δ₂):**
```
δ₂ = ∂L/∂z₂ = ∂L/∂ŷ × ∂ŷ/∂z₂
   = -1.6145 × 0.2357
   = -0.3806
```

**Simplified formula** (for sigmoid + cross-entropy):
```
δ₂ = ŷ - y = 0.6194 - 1 = -0.3806 ✓
```

### Step 3.2: Gradients for W₂ and b₂

**Weight gradient:**
```
∂L/∂W₂ = δ₂ × h

∂L/∂W₂[0] = -0.3806 × 0.21 = -0.0799
∂L/∂W₂[1] = -0.3806 × 0.47 = -0.1789
```

**Bias gradient:**
```
∂L/∂b₂ = δ₂ = -0.3806
```

### Step 3.3: Backpropagate to Hidden Layer

**Gradient w.r.t. hidden output:**
```
∂L/∂h = W₂ᵀ × δ₂

∂L/∂h[0] = 0.5 × (-0.3806) = -0.1903
∂L/∂h[1] = 0.6 × (-0.3806) = -0.2284
```

**Gradient through ReLU:**
```
ReLU'(z) = { 1 if z > 0
           { 0 if z ≤ 0
```

Since z₁ = [0.21, 0.47], both > 0:
```
δ₁ = ∂L/∂h ⊙ ReLU'(z₁)

δ₁[0] = -0.1903 × 1 = -0.1903
δ₁[1] = -0.2284 × 1 = -0.2284
```

**Result:** `δ₁ = [-0.1903, -0.2284]`

### Step 3.4: Gradients for W₁ and b₁

**Weight gradient:**
```
∂L/∂W₁ = δ₁ ⊗ x (outer product)

∂L/∂W₁ = | δ₁[0]×x[0]  δ₁[0]×x[1] |
         | δ₁[1]×x[0]  δ₁[1]×x[1] |

       = | -0.1903×0.5  -0.1903×0.3 |
         | -0.2284×0.5  -0.2284×0.3 |

       = | -0.0952  -0.0571 |
         | -0.1142  -0.0685 |
```

**Bias gradient:**
```
∂L/∂b₁ = δ₁ = [-0.1903, -0.2284]
```

---

## Weight Updates

**Update rule:**
```
θ_new = θ_old - η × ∂L/∂θ
```

### Update W₂:
```
W₂_new = W₂ - η × ∂L/∂W₂
       = [0.5, 0.6] - 0.1 × [-0.0799, -0.1789]
       = [0.5 + 0.00799, 0.6 + 0.01789]
       = [0.508, 0.618]
```

### Update b₂:
```
b₂_new = 0.1 - 0.1 × (-0.3806) = 0.138
```

### Update W₁:
```
W₁_new = W₁ - 0.1 × ∂L/∂W₁

       = | 0.1 + 0.00952   0.2 + 0.00571 |
         | 0.3 + 0.01142   0.4 + 0.00685 |

       = | 0.110   0.206 |
         | 0.311   0.407 |
```

### Update b₁:
```
b₁_new = [0.1, 0.2] - 0.1 × [-0.1903, -0.2284]
       = [0.119, 0.223]
```

### Summary of Updates

| Parameter | Before | After |
|-----------|--------|-------|
| W₁[0,0] | 0.100 | 0.110 |
| W₁[0,1] | 0.200 | 0.206 |
| W₁[1,0] | 0.300 | 0.311 |
| W₁[1,1] | 0.400 | 0.407 |
| b₁ | [0.1, 0.2] | [0.119, 0.223] |
| W₂ | [0.5, 0.6] | [0.508, 0.618] |
| b₂ | 0.1 | 0.138 |

**Key Observation:** All weights increased because the prediction (0.62) was lower than the target (1.0), so the network adjusted to produce higher outputs.

---

## Why We Use Matrices

### 1. Compact Representation

**Without Matrices (Scalar Form):**
```
z₁[0] = w₁₀₀·x[0] + w₁₀₁·x[1] + b₁[0]
z₁[1] = w₁₁₀·x[0] + w₁₁₁·x[1] + b₁[1]
```

**With Matrices (Vectorized Form):**
```
z₁ = W₁·x + b₁
```

One equation replaces multiple scalar equations.

### 2. Matrix Multiplication Mechanics

**The Weight Matrix Structure:**
```
W₁ = | 0.1   0.2 |    (2×2 matrix)
     | 0.3   0.4 |

x = | 0.5 |           (2×1 vector)
    | 0.3 |
```

**Matrix-Vector Multiplication:**
```
(W·x)[i] = Σⱼ Wᵢⱼ·xⱼ
```

**Step-by-step:**

Row 1 of W₁ × x:
```
z₁[0] = [0.1, 0.2] · [0.5, 0.3]ᵀ
      = 0.1×0.5 + 0.2×0.3
      = 0.05 + 0.06 = 0.11
```

Row 2 of W₁ × x:
```
z₁[1] = [0.3, 0.4] · [0.5, 0.3]ᵀ
      = 0.3×0.5 + 0.4×0.3
      = 0.15 + 0.12 = 0.27
```

### 3. Matrix Structure Interpretation

Each **row** of W₁ represents **weights for one neuron**:
- Row 1: weights for neuron 1 in hidden layer
- Row 2: weights for neuron 2 in hidden layer

Each **column** of W₁ represents **connections from one input**:
- Column 1: how x[0] affects all hidden neurons
- Column 2: how x[1] affects all hidden neurons

### 4. Computational Efficiency

**Parallel Processing:**

Modern GPUs can perform matrix operations in parallel:

- **Scalar approach:** 500,000 sequential operations (for 1000×500 layer)
- **Matrix approach:** One highly parallelized operation
- **Speedup:** 10-100x faster on GPUs

### 5. Key Matrix Operations

#### Matrix-Vector Multiplication (Forward Pass)
```
z = W·x
```
**Use:** Computing neuron activations

#### Transpose (Backpropagation)
```
∂L/∂h = W₂ᵀ · δ₂
```
**Use:** Passing gradients backward through layers

#### Outer Product (Gradient Calculation)
```
∂L/∂W = δ ⊗ xᵀ
```
**Use:** Computing weight gradients

**Example:**
```
δ₁ = | -0.1903 |       x = [0.5, 0.3]
     | -0.2284 |

∂L/∂W₁ = | -0.1903 | ⊗ [0.5, 0.3]
         | -0.2284 |

       = | -0.0952  -0.0571 |
         | -0.1142  -0.0685 |
```

#### Element-wise Multiplication (Hadamard Product)
```
δ₁ = ∂L/∂h ⊙ ReLU'(z₁)
```
**Symbol:** ⊙  
**Use:** Applying activation derivatives

### 6. Dimension Consistency

For a network: `n_input → n_hidden → n_output`

**Weights must satisfy:**
```
W₁: (n_hidden × n_input)
W₂: (n_output × n_hidden)
```

In our example: `2 → 2 → 1`
```
W₁: (2×2)  - connects 2 inputs to 2 hidden neurons
W₂: (1×2)  - connects 2 hidden neurons to 1 output
```

**Matrix Multiplication Rules:**

For `C = A·B` to be valid:
```
A: (m × n)
B: (n × p)
C: (m × p)
```

Inner dimensions must match (n = n)

### 7. Comparison Summary

| Aspect | Scalar Approach | Matrix Approach |
|--------|----------------|-----------------|
| **Notation** | Many equations | Compact |
| **Speed** | Sequential | Parallel |
| **Scalability** | Poor | Excellent |
| **Hardware** | CPU-bound | GPU-optimized |
| **Code complexity** | High | Low |

---

## Key Takeaways

### Deep Learning Core Concepts

1. **Chain Rule**: The backbone of backpropagation
   ```
   ∂L/∂W = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂W
   ```

2. **Activation Functions**:
   - **ReLU:** Simple gradient (0 or 1), prevents vanishing gradients
   - **Sigmoid:** Derivative = σ(z)(1-σ(z)), squashes to [0,1]

3. **Gradient Flow**: Gradients flow backward through the network
   ```
   Loss → Output → Hidden → Input
   ```

4. **Why Deep Learning Works**:
   - **Automatic feature learning:** Hidden layers learn useful representations
   - **Gradient descent:** Iteratively minimizes loss
   - **Composability:** Stacking layers enables learning complex functions

### Matrix Benefits

- **Scalability:** Modern networks have millions of parameters; matrices make this manageable
- **Automatic Differentiation:** Frameworks (PyTorch, TensorFlow) use matrices for automatic gradient computation
- **Hardware Optimization:** GPUs are designed for matrix operations, enabling 100-1000x speedups

### Training Process

This forward-backward-update process repeats for thousands of iterations across many training examples until the network converges to optimal weights.

---

**Note:** This example demonstrates one complete training iteration. In practice, deep learning involves:
- Thousands to millions of iterations
- Batch processing (multiple examples at once)
- More sophisticated optimizers (Adam, RMSprop)
- Regularization techniques (dropout, weight decay)
- Multiple hidden layers with hundreds/thousands of neurons
