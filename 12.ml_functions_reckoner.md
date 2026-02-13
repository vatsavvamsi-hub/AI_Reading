# Machine Learning Functions Reckoner

## Activation Functions

### 1. Sigmoid (Logistic)
**Formula:**
```
σ(x) = 1 / (1 + e^(-x))
```
**Output Range:** (0, 1)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Binary classification (output layer) | Smooth gradient | Vanishing gradient problem |
| Probability outputs | Output bounded (0,1) | Not zero-centered |
| Logistic regression | | Computationally expensive (exp) |

---

### 2. Tanh (Hyperbolic Tangent)
**Formula:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
**Output Range:** (-1, 1)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Hidden layers (older networks) | Zero-centered | Vanishing gradient problem |
| RNNs | Stronger gradients than sigmoid | Computationally expensive |
| When negative outputs needed | | |

---

### 3. ReLU (Rectified Linear Unit)
**Formula:**
```
f(x) = max(0, x)
```
**Output Range:** [0, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Default for hidden layers | Computationally efficient | Dying ReLU (neurons stuck at 0) |
| CNNs | No vanishing gradient (positive) | Not zero-centered |
| Deep networks | Sparse activation | Unbounded output |

---

### 4. Leaky ReLU
**Formula:**
```
f(x) = x       if x > 0
f(x) = αx     if x ≤ 0    (α typically 0.01)
```
**Output Range:** (-∞, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| When dying ReLU is a problem | Prevents dying ReLU | α is a hyperparameter |
| Deep networks | Allows negative values | Inconsistent results |
| GANs | | |

---

### 5. Parametric ReLU (PReLU)
**Formula:**
```
f(x) = x       if x > 0
f(x) = αx     if x ≤ 0    (α is learnable)
```
**Output Range:** (-∞, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Image classification | α learned during training | Risk of overfitting |
| When Leaky ReLU α is suboptimal | Adaptive | More parameters |

---

### 6. ELU (Exponential Linear Unit)
**Formula:**
```
f(x) = x              if x > 0
f(x) = α(e^x - 1)     if x ≤ 0    (α typically 1.0)
```
**Output Range:** (-α, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Deep networks | Smooth for negative values | Computationally expensive |
| When mean activations near zero | Reduces bias shift | α is hyperparameter |
| | No dying ReLU | |

---

### 7. SELU (Scaled ELU)
**Formula:**
```
f(x) = λ * x              if x > 0
f(x) = λ * α(e^x - 1)     if x ≤ 0

λ ≈ 1.0507, α ≈ 1.6733
```
**Output Range:** (-λα, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Self-normalizing networks | Self-normalizing property | Requires specific initialization |
| Fully connected deep networks | No need for batch norm | Only works with specific architectures |

---

### 8. Softmax
**Formula:**
```
softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)
```
**Output Range:** (0, 1), sum = 1

| Use Cases | Pros | Cons |
|-----------|------|------|
| Multi-class classification (output) | Outputs probability distribution | Computationally expensive |
| Attention mechanisms | Differentiable argmax | Sensitive to outliers |
| Neural machine translation | | |

---

### 9. Swish / SiLU
**Formula:**
```
f(x) = x * σ(x) = x / (1 + e^(-x))
```
**Output Range:** (-0.278, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Deep networks | Smooth, non-monotonic | More computation than ReLU |
| Transformer models | Outperforms ReLU in deep nets | |
| EfficientNet, GPT | Self-gated | |

---

### 10. GELU (Gaussian Error Linear Unit)
**Formula:**
```
f(x) = x * Φ(x)
Φ(x) = 0.5 * (1 + erf(x/√2))

Approximation:
f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```
**Output Range:** (-0.17, ∞)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Transformers (BERT, GPT) | Smooth | Computationally expensive |
| NLP models | Probabilistic interpretation | |
| Vision Transformers | State-of-the-art results | |

---

## Loss Functions

### Regression Losses

#### 1. Mean Squared Error (MSE) / L2 Loss
**Formula:**
```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| General regression | Differentiable everywhere | Sensitive to outliers |
| When large errors are bad | Penalizes large errors heavily | Squared units |
| Gaussian noise assumption | Easy optimization | |

---

#### 2. Mean Absolute Error (MAE) / L1 Loss
**Formula:**
```
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Robust regression | Robust to outliers | Not differentiable at 0 |
| When outliers present | Same units as target | Harder to optimize |
| Median estimation | | |

---

#### 3. Huber Loss (Smooth L1)
**Formula:**
```
L_δ(y, ŷ) = 0.5(y - ŷ)²           if |y - ŷ| ≤ δ
L_δ(y, ŷ) = δ|y - ŷ| - 0.5δ²      if |y - ŷ| > δ
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Regression with outliers | Best of MSE and MAE | δ is a hyperparameter |
| Object detection (bbox) | Differentiable everywhere | |
| Robust regression | Less sensitive to outliers | |

---

#### 4. Log-Cosh Loss
**Formula:**
```
L(y, ŷ) = Σᵢ log(cosh(ŷᵢ - yᵢ))
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Robust regression | Smooth, twice differentiable | Less interpretable |
| Alternative to Huber | Similar to MSE for small errors | |
| | Similar to MAE for large errors | |

---

### Classification Losses

#### 5. Binary Cross-Entropy (Log Loss)
**Formula:**
```
BCE = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Binary classification | Probabilistic interpretation | Requires sigmoid output |
| Multi-label classification | Heavily penalizes confident wrong | Sensitive to class imbalance |
| Output layer with sigmoid | | |

---

#### 6. Categorical Cross-Entropy
**Formula:**
```
CCE = -Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
```
(where j is class index, i is sample)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Multi-class classification | Standard for classification | Requires one-hot encoding |
| Output layer with softmax | Well-calibrated probabilities | Sensitive to class imbalance |
| Neural networks | | |

---

#### 7. Sparse Categorical Cross-Entropy
**Formula:**
```
SCCE = -(1/n) Σᵢ log(ŷᵢ,yᵢ)
```
(yᵢ is integer class label)

| Use Cases | Pros | Cons |
|-----------|------|------|
| Multi-class (integer labels) | Memory efficient | Same as CCE |
| Large number of classes | No one-hot encoding needed | |

---

#### 8. Focal Loss
**Formula:**
```
FL = -αₜ(1 - pₜ)^γ log(pₜ)

where pₜ = p if y=1, else 1-p
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Highly imbalanced datasets | Down-weights easy examples | γ, α are hyperparameters |
| Object detection | Focuses on hard examples | |
| RetinaNet | Handles class imbalance | |

---

#### 9. Hinge Loss
**Formula:**
```
L = (1/n) Σᵢ max(0, 1 - yᵢ * ŷᵢ)

(y ∈ {-1, +1})
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| SVM classification | Maximum margin classifier | Not differentiable at 1 |
| Binary classification | Robust to outliers | Only for binary |
| | | No probability output |

---

#### 10. Squared Hinge Loss
**Formula:**
```
L = (1/n) Σᵢ max(0, 1 - yᵢ * ŷᵢ)²
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| SVM with smoother gradients | Differentiable | Penalizes violations more |
| When hinge is too harsh | Easier optimization | |

---

### Specialized Losses

#### 11. KL Divergence (Kullback-Leibler)
**Formula:**
```
D_KL(P || Q) = Σᵢ P(i) log(P(i) / Q(i))
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| VAEs (variational inference) | Measures distribution difference | Asymmetric |
| Knowledge distillation | Information theoretic | Undefined if Q(i)=0 |
| Probability distribution matching | | |

---

#### 12. Contrastive Loss
**Formula:**
```
L = (1-y) * 0.5 * D² + y * 0.5 * max(0, m - D)²

D = ||f(x₁) - f(x₂)||
y = 0 (similar), 1 (dissimilar)
m = margin
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Siamese networks | Learns similarity | Requires pairs |
| Face verification | Embedding learning | Margin tuning |
| One-shot learning | | |

---

#### 13. Triplet Loss
**Formula:**
```
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)

a = anchor, p = positive, n = negative
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Face recognition | Better than contrastive | Hard negative mining needed |
| Image retrieval | Relative distances | Triplet sampling complexity |
| Metric learning | | |

---

#### 14. CTC Loss (Connectionist Temporal Classification)
**Formula:**
```
L = -log P(y | x) = -log Σ_π∈B^(-1)(y) P(π | x)
```

| Use Cases | Pros | Cons |
|-----------|------|------|
| Speech recognition | Handles variable-length output | Computationally expensive |
| OCR | No alignment needed | Assumes independence |
| Sequence-to-sequence | | |

---

## Quick Reference: Choosing Functions

### Activation Function Selection
| Task | Recommended |
|------|-------------|
| Hidden layers (default) | ReLU, Leaky ReLU |
| Deep networks | GELU, Swish, SELU |
| Transformers/NLP | GELU |
| Binary output | Sigmoid |
| Multi-class output | Softmax |
| RNN/LSTM | Tanh (gates), Sigmoid |
| GANs | Leaky ReLU |

### Loss Function Selection
| Task | Recommended |
|------|-------------|
| Regression (no outliers) | MSE |
| Regression (with outliers) | MAE, Huber |
| Binary classification | Binary Cross-Entropy |
| Multi-class classification | Categorical Cross-Entropy |
| Imbalanced classification | Focal Loss |
| Embedding/Similarity learning | Triplet Loss, Contrastive |
| Distribution matching | KL Divergence |
| Sequence labeling | CTC Loss |
