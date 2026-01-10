# Machine Learning Mathematics Tutorial
## From Foundations to Backpropagation

---

## Table of Contents
1. [Linear Algebra](#1-linear-algebra)
2. [Calculus](#2-calculus)
3. [Probability](#3-probability)
4. [Logarithms](#4-logarithms)
5. [Activation Functions](#5-activation-functions)
6. [Backpropagation](#6-backpropagation)

---

## 1. Linear Algebra

Linear algebra is the foundation of machine learning, enabling us to work with high-dimensional data efficiently.

### 1.1 Vectors

**Definition**: A vector is an ordered array of numbers.

**Example 1**: Feature vector for a house
```
x = [2000, 3, 2, 1]  # [square_feet, bedrooms, bathrooms, garage]
```

**Operations**:
- **Addition**: `u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]`
- **Scalar multiplication**: `c·v = [c·v₁, c·v₂, ..., c·vₙ]`
- **Dot product**: `u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ`

**Example 2**: Computing similarity
```
user1_preferences = [5, 3, 0, 4]  # ratings for [action, comedy, horror, drama]
user2_preferences = [4, 4, 1, 3]

similarity = user1_preferences · user2_preferences = 5×4 + 3×4 + 0×1 + 4×3 = 44
```

**Magnitude (Norm)**:
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**Example 3**: Normalizing a feature vector
```
v = [3, 4]
||v|| = √(3² + 4²) = √25 = 5
normalized_v = v/||v|| = [3/5, 4/5] = [0.6, 0.8]
```

### 1.2 Matrices

**Definition**: A matrix is a 2D array of numbers with m rows and n columns.

**Example 4**: Dataset representation
```
X = [[1500, 2, 1],    # house 1: [sqft, bedrooms, bathrooms]
     [2000, 3, 2],    # house 2
     [1800, 3, 1.5]]  # house 3
```

**Matrix Multiplication**: For matrices A(m×n) and B(n×p), result C(m×p):
```
C[i,j] = Σ(k=1 to n) A[i,k] × B[k,j]
```

**Example 5**: Neural network layer computation
```
Input: x = [x₁, x₂]ᵀ
Weights: W = [[w₁₁, w₁₂],
              [w₂₁, w₂₂],
              [w₃₁, w₃₂]]
Bias: b = [b₁, b₂, b₃]ᵀ

Output: z = Wx + b

If x = [2, 3]ᵀ, W = [[0.5, 0.3], [0.2, 0.8], [0.1, 0.4]], b = [0.1, 0.2, 0.3]ᵀ
z = [[0.5×2 + 0.3×3], [0.2×2 + 0.8×3], [0.1×2 + 0.4×3]] + [0.1, 0.2, 0.3]ᵀ
z = [1.9, 2.8, 1.5]ᵀ
```

**Transpose**: Flip rows and columns
```
A = [[1, 2, 3],      Aᵀ = [[1, 4],
     [4, 5, 6]]            [2, 5],
                           [3, 6]]
```

**Example 6**: Computing covariance matrix
```
X = [[1, 2],    # 3 samples, 2 features
     [2, 4],
     [3, 5]]

Centered X: X_centered = X - mean(X)
Covariance: Σ = (1/n) × Xᵀ × X
```

### 1.3 Matrix Properties in ML

**Identity Matrix** (I): Diagonal matrix of ones
```
I₃ = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]]
```

**Inverse Matrix** (A⁻¹): Used in solving linear systems
```
If A·A⁻¹ = I, then x = A⁻¹b solves Ax = b
```

**Example 7**: Normal equation for linear regression
```
θ = (XᵀX)⁻¹Xᵀy

Where:
- X is the design matrix (features)
- y is the target vector
- θ is the parameter vector we're solving for
```

---

## 2. Calculus

Calculus enables us to optimize machine learning models by finding parameter values that minimize loss.

### 2.1 Derivatives

**Definition**: The derivative measures the rate of change of a function.

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

**Example 8**: Simple derivatives
```
f(x) = x²        →  f'(x) = 2x
f(x) = 3x³       →  f'(x) = 9x²
f(x) = eˣ        →  f'(x) = eˣ
f(x) = ln(x)     →  f'(x) = 1/x
```

**Derivative Rules**:
- **Sum rule**: `(f + g)' = f' + g'`
- **Product rule**: `(f·g)' = f'g + fg'`
- **Chain rule**: `(f(g(x)))' = f'(g(x))·g'(x)`

**Example 9**: Chain rule application
```
f(x) = (3x² + 2)⁵

Let u = 3x² + 2, then f = u⁵
df/dx = df/du × du/dx = 5u⁴ × 6x = 5(3x² + 2)⁴ × 6x = 30x(3x² + 2)⁴
```

### 2.2 Partial Derivatives

**Definition**: Derivative with respect to one variable, holding others constant.

**Example 10**: Loss function for linear regression
```
L(w, b) = (1/2n) Σᵢ(wxᵢ + b - yᵢ)²

∂L/∂w = (1/n) Σᵢ(wxᵢ + b - yᵢ)xᵢ
∂L/∂b = (1/n) Σᵢ(wxᵢ + b - yᵢ)
```

**Example 11**: Numerical example
```
Given: L(w,b) = (wx + b - y)² with x=2, y=5, w=1, b=1
L(1,1) = (1×2 + 1 - 5)² = (-2)² = 4

∂L/∂w = 2(wx + b - y)×x = 2(1×2 + 1 - 5)×2 = 2(-2)×2 = -8
∂L/∂b = 2(wx + b - y)×1 = 2(1×2 + 1 - 5)×1 = 2(-2)×1 = -4
```

### 2.3 Gradient

**Definition**: Vector of all partial derivatives.

```
∇f(x₁, x₂, ..., xₙ) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Example 12**: Gradient for 2D function
```
f(x, y) = x² + 2y²

∇f = [∂f/∂x, ∂f/∂y] = [2x, 4y]

At point (3, 1): ∇f(3,1) = [6, 4]
This means: moving in direction [6, 4] increases f most rapidly
```

### 2.4 Gradient Descent

**Algorithm**: Iteratively update parameters in the opposite direction of the gradient.

```
θₙₑw = θₒₗd - α∇L(θₒₗd)

Where:
- α is the learning rate
- ∇L is the gradient of the loss function
```

**Example 13**: One step of gradient descent
```
Loss: L(w) = (w - 3)²
Gradient: ∂L/∂w = 2(w - 3)

Initial: w₀ = 0, α = 0.1
∂L/∂w|ᵨ₌₀ = 2(0 - 3) = -6
w₁ = w₀ - α × (-6) = 0 - 0.1 × (-6) = 0.6

w₂: ∂L/∂w|ᵨ₌₀.₆ = 2(0.6 - 3) = -4.8
w₂ = 0.6 - 0.1 × (-4.8) = 1.08

Converging toward w=3 (the minimum)
```

**Example 14**: Multi-variable gradient descent
```
L(w₁, w₂) = w₁² + 4w₂²
∇L = [2w₁, 8w₂]

Start: (w₁, w₂) = (4, 2), α = 0.1
∇L(4, 2) = [8, 16]

Update:
w₁ = 4 - 0.1×8 = 3.2
w₂ = 2 - 0.1×16 = 0.4
```

---

## 3. Probability

Probability theory helps us model uncertainty and make predictions.

### 3.1 Basic Probability

**Definitions**:
- **P(A)**: Probability of event A, ranges from 0 to 1
- **P(A ∪ B)**: Probability of A OR B
- **P(A ∩ B)**: Probability of A AND B

**Rules**:
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(Aᶜ) = 1 - P(A)  # complement
```

**Example 15**: Email classification
```
P(Spam) = 0.3
P(Contains "lottery") = 0.05
P(Contains "lottery" | Spam) = 0.4

What's P(Spam AND Contains "lottery")?
P(Spam ∩ Lottery) = P(Lottery | Spam) × P(Spam) = 0.4 × 0.3 = 0.12
```

### 3.2 Conditional Probability

**Definition**: Probability of A given that B has occurred.

```
P(A|B) = P(A ∩ B) / P(B)
```

**Example 16**: Medical diagnosis
```
P(Disease) = 0.01          # 1% have the disease
P(+Test | Disease) = 0.99  # 99% sensitivity
P(+Test | ¬Disease) = 0.05 # 5% false positive

If test is positive, what's P(Disease | +Test)?

Using Bayes' theorem (see below)
```

### 3.3 Bayes' Theorem

**Formula**: Updates beliefs based on evidence.

```
P(A|B) = [P(B|A) × P(A)] / P(B)
```

**Example 17**: Spam filter (continued from Example 15)
```
P(Spam | Lottery) = [P(Lottery | Spam) × P(Spam)] / P(Lottery)
                  = (0.4 × 0.3) / 0.05
                  = 0.12 / 0.05
                  = 2.4... wait, this exceeds 1!

Actually, we need: P(Lottery) = P(L|S)P(S) + P(L|¬S)P(¬S)
Assuming P(Lottery|¬Spam) = 0.01:
P(Lottery) = 0.4×0.3 + 0.01×0.7 = 0.12 + 0.007 = 0.127

P(Spam | Lottery) = 0.12 / 0.127 ≈ 0.945 (94.5% spam)
```

### 3.4 Probability Distributions

**Bernoulli Distribution**: Single binary trial
```
P(X=1) = p
P(X=0) = 1-p

Example: Coin flip with p=0.6 for heads
```

**Normal (Gaussian) Distribution**:
```
f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

Where:
- μ is the mean
- σ² is the variance
```

**Example 18**: Assuming height follows N(170, 10²) cm
```
μ = 170 cm
σ = 10 cm

P(160 < height < 180) ≈ 0.68 (68% within 1 std dev)
P(150 < height < 190) ≈ 0.95 (95% within 2 std dev)
```

### 3.5 Expected Value

**Definition**: Average value weighted by probability.

```
E[X] = Σ xᵢ × P(X=xᵢ)
```

**Example 19**: Expected loss in binary classification
```
Outcomes: Correct (+10 points), Wrong (-5 points)
P(Correct) = 0.8, P(Wrong) = 0.2

E[Score] = 10×0.8 + (-5)×0.2 = 8 - 1 = 7 points
```

**Example 20**: Expected value in regression
```
Model predicts ŷ, true value is y

Expected squared error: E[(y - ŷ)²]

This is what we minimize in mean squared error (MSE) loss!
```

---

## 4. Logarithms

Logarithms transform multiplicative relationships into additive ones, essential for many ML algorithms.

### 4.1 Logarithm Basics

**Definition**: `log_b(x) = y` means `b^y = x`

**Common bases**:
- `log₁₀(x)`: Base 10 (common logarithm)
- `ln(x) = log_e(x)`: Natural logarithm (base e ≈ 2.718)
- `log₂(x)`: Base 2 (used in information theory)

**Example 21**: Basic logarithms
```
log₁₀(100) = 2     because 10² = 100
log₂(8) = 3        because 2³ = 8
ln(e²) = 2         because e² = e²
ln(1) = 0          because e⁰ = 1
```

### 4.2 Logarithm Properties

**Key properties**:
```
1. log(ab) = log(a) + log(b)
2. log(a/b) = log(a) - log(b)
3. log(aⁿ) = n·log(a)
4. log(1) = 0
5. log_b(b) = 1
```

**Example 22**: Simplifying expressions
```
log(x²y³/z) = log(x²) + log(y³) - log(z)
            = 2log(x) + 3log(y) - log(z)
```

### 4.3 Logarithms in Machine Learning

**Use Case 1: Numerical Stability**

When multiplying many small probabilities, use log space:

**Example 23**: Naive Bayes probability
```
P(Class | Features) ∝ P(f₁|C) × P(f₂|C) × ... × P(fₙ|C) × P(C)

If each probability ≈ 0.1, with n=100:
Direct: (0.1)¹⁰⁰ ≈ 10⁻¹⁰⁰ (underflow!)

Log space:
log P = log(P(f₁|C)) + log(P(f₂|C)) + ... + log(P(C))
      = 100 × log(0.1) = 100 × (-2.303) = -230.3
Much more stable!
```

**Use Case 2: Loss Functions**

**Example 24**: Binary cross-entropy loss
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

For true label y=1, prediction ŷ=0.9:
L = -[1×log(0.9) + 0×log(0.1)]
  = -log(0.9)
  ≈ -(-0.105)
  = 0.105

For ŷ=0.1 (wrong):
L = -log(0.1) ≈ 2.303 (much higher penalty)
```

**Use Case 3: Feature Scaling**

**Example 25**: Log transform for skewed features
```
Original: Income = [20000, 25000, 30000, 500000, 1000000]
Log transform: log(Income) = [9.90, 10.13, 10.31, 13.12, 13.82]

Reduces the scale and makes distribution more normal
```

### 4.4 Exponential and Logarithm Relationship

**Inverse relationship**:
```
If y = log(x), then x = eʸ
If y = eˣ, then x = ln(y)
```

**Example 26**: Converting log predictions back
```
Model predicts: log(price) = 12.5
Actual price: price = e¹²·⁵ ≈ 268,337

This is useful when training on log-transformed targets!
```

---

## 5. Activation Functions

Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.

### 5.1 Why Non-linearity Matters

**Without activation functions**:
```
Layer 1: z₁ = W₁x + b₁
Layer 2: z₂ = W₂z₁ + b₂ = W₂(W₁x + b₁) + b₂ = W₂W₁x + W₂b₁ + b₂

This is just a linear transformation! Multiple layers = one layer.
```

**With activation functions**:
```
Layer 1: z₁ = W₁x + b₁, a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂, a₂ = σ(z₂)

Now the network can learn non-linear patterns!
```

### 5.2 Common Activation Functions

#### Sigmoid

**Formula**:
```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Properties**:
- Output range: (0, 1)
- S-shaped curve
- Smooth and differentiable

**Example 27**: Sigmoid values
```
σ(0) = 1/(1 + e⁰) = 1/(1 + 1) = 0.5
σ(2) = 1/(1 + e⁻²) = 1/(1 + 0.135) ≈ 0.881
σ(-2) = 1/(1 + e²) = 1/(1 + 7.389) ≈ 0.119
σ(10) ≈ 0.9999 (approaches 1)
σ(-10) ≈ 0.0001 (approaches 0)
```

**Derivative**:
```
σ'(x) = σ(x)(1 - σ(x))
```

**Example 28**: Sigmoid gradient
```
At x=0: σ(0)=0.5, σ'(0) = 0.5×(1-0.5) = 0.25
At x=2: σ(2)≈0.881, σ'(2) ≈ 0.881×0.119 ≈ 0.105
At x=5: σ(5)≈0.993, σ'(5) ≈ 0.993×0.007 ≈ 0.007 (vanishing!)
```

**Use case**: Binary classification output layer

#### Hyperbolic Tangent (tanh)

**Formula**:
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) - 1
```

**Properties**:
- Output range: (-1, 1)
- Zero-centered (advantage over sigmoid)
- S-shaped curve

**Example 29**: Tanh values
```
tanh(0) = 0
tanh(1) ≈ 0.762
tanh(-1) ≈ -0.762
tanh(3) ≈ 0.995
tanh(-3) ≈ -0.995
```

**Derivative**:
```
tanh'(x) = 1 - tanh²(x)
```

**Example 30**: Tanh gradient
```
At x=0: tanh'(0) = 1 - 0² = 1 (maximum gradient)
At x=2: tanh'(2) = 1 - 0.964² ≈ 0.071
```

#### ReLU (Rectified Linear Unit)

**Formula**:
```
ReLU(x) = max(0, x) = {x if x > 0; 0 if x ≤ 0}
```

**Properties**:
- Output range: [0, ∞)
- Non-saturating for positive values
- Computationally efficient
- Most popular for hidden layers

**Example 31**: ReLU values
```
ReLU(-5) = 0
ReLU(-0.1) = 0
ReLU(0) = 0
ReLU(0.1) = 0.1
ReLU(5) = 5
ReLU(100) = 100
```

**Derivative**:
```
ReLU'(x) = {1 if x > 0; 0 if x ≤ 0}
```

**Example 32**: ReLU in a network
```
Input: x = [-2, 1, 3]
Weights: W = [[0.5], [1.0], [-0.5]]
Bias: b = 0.5

z = Wx + b = 0.5×(-2) + 1.0×1 + (-0.5)×3 + 0.5 = -1.5
a = ReLU(z) = ReLU(-1.5) = 0

All negative pre-activations become 0!
```

#### Leaky ReLU

**Formula**:
```
LeakyReLU(x) = {x if x > 0; αx if x ≤ 0}

Common: α = 0.01
```

**Properties**:
- Solves "dying ReLU" problem
- Small gradient for negative values

**Example 33**: Leaky ReLU comparison
```
x = [-2, -1, 0, 1, 2]

ReLU(x) = [0, 0, 0, 1, 2]
LeakyReLU(x, α=0.01) = [-0.02, -0.01, 0, 1, 2]

Negative inputs still contribute small gradients!
```

#### Softmax

**Formula**: Converts logits to probabilities
```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

**Properties**:
- Output sums to 1
- Used for multi-class classification
- Emphasizes largest values

**Example 34**: Softmax computation
```
Logits: z = [2.0, 1.0, 0.1]

e^z = [e², e¹, e⁰·¹] = [7.389, 2.718, 1.105]
Sum = 7.389 + 2.718 + 1.105 = 11.212

softmax(z) = [7.389/11.212, 2.718/11.212, 1.105/11.212]
           = [0.659, 0.242, 0.099]

Class 0 has 65.9% probability!
```

**Example 35**: Softmax with larger differences
```
z = [5.0, 2.0, 0.1]

e^z = [148.41, 7.389, 1.105]
Sum = 156.904

softmax(z) = [0.946, 0.047, 0.007]

Large differences in logits → high confidence prediction
```

### 5.3 Choosing Activation Functions

**Guidelines**:
```
- Hidden layers: ReLU (default choice)
- Binary classification output: Sigmoid
- Multi-class classification output: Softmax
- Regression output: Linear (no activation)
- Recurrent networks: tanh or sigmoid
```

---

## 6. Backpropagation

Backpropagation efficiently computes gradients for training neural networks using the chain rule.

### 6.1 The Problem

**Forward pass**: Compute predictions
**Challenge**: Compute ∂L/∂w for every weight w in the network

**Naive approach**: Compute each gradient independently → O(n²) complexity
**Backpropagation**: Reuse computations → O(n) complexity

### 6.2 Simple Example: One Neuron

**Network**:
```
Input: x
Weight: w
Bias: b
Pre-activation: z = wx + b
Activation: a = σ(z)
Loss: L = (a - y)²
```

**Example 36**: Forward pass
```
Given: x=2, y=1, w=0.5, b=0.1

z = wx + b = 0.5×2 + 0.1 = 1.1
a = σ(z) = 1/(1+e⁻¹·¹) ≈ 0.750
L = (a - y)² = (0.750 - 1)² = 0.0625
```

**Backward pass** (chain rule):
```
∂L/∂a = 2(a - y) = 2(0.750 - 1) = -0.500
∂a/∂z = σ'(z) = σ(z)(1-σ(z)) = 0.750×0.250 = 0.1875
∂z/∂w = x = 2
∂z/∂b = 1

∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w = -0.500 × 0.1875 × 2 = -0.1875
∂L/∂b = ∂L/∂a × ∂a/∂z × ∂z/∂b = -0.500 × 0.1875 × 1 = -0.09375
```

**Update** (gradient descent with α=0.1):
```
w_new = w - α × ∂L/∂w = 0.5 - 0.1×(-0.1875) = 0.51875
b_new = b - α × ∂L/∂b = 0.1 - 0.1×(-0.09375) = 0.109375
```

### 6.3 Two-Layer Network

**Architecture**:
```
Input layer: x (n inputs)
Hidden layer: h = σ(W₁x + b₁) (m neurons)
Output layer: ŷ = σ(W₂h + b₂) (1 neuron)
Loss: L = (ŷ - y)²
```

**Example 37**: Concrete two-layer network
```
x = [1, 2]ᵀ
W₁ = [[0.5, 0.3],    # 2×2 matrix
      [0.2, 0.4]]
b₁ = [0.1, 0.2]ᵀ
W₂ = [[0.6, 0.5]]    # 1×2 matrix
b₂ = [0.1]
y = 1 (true label)
```

**Forward pass**:
```
Step 1: Hidden layer pre-activation
z₁ = W₁x + b₁ = [[0.5×1 + 0.3×2],  + [0.1]  = [1.2]
                 [0.2×1 + 0.4×2]]    [0.2]    [1.2]

Step 2: Hidden layer activation (using sigmoid)
h = σ(z₁) = [σ(1.2), σ(1.2)] = [0.769, 0.769]

Step 3: Output layer pre-activation
z₂ = W₂h + b₂ = [0.6×0.769 + 0.5×0.769] + 0.1 = 0.846 + 0.1 = 0.946

Step 4: Output activation
ŷ = σ(z₂) = σ(0.946) = 0.720

Step 5: Loss
L = (ŷ - y)² = (0.720 - 1)² = 0.0784
```

**Backward pass**:
```
Step 1: Output layer gradient
∂L/∂ŷ = 2(ŷ - y) = 2(0.720 - 1) = -0.560
∂ŷ/∂z₂ = σ'(z₂) = 0.720×(1-0.720) = 0.202
δ₂ = ∂L/∂z₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ = -0.560 × 0.202 = -0.113

Step 2: Output layer weight gradients
∂L/∂W₂ = δ₂ × hᵀ = -0.113 × [0.769, 0.769] = [-0.087, -0.087]
∂L/∂b₂ = δ₂ = -0.113

Step 3: Hidden layer gradient (backpropagate through W₂)
∂L/∂h = W₂ᵀ × δ₂ = [[0.6],  × [-0.113] = [-0.068]
                     [0.5]]              [-0.057]

Step 4: Hidden layer pre-activation
∂h/∂z₁ = σ'(z₁) = [0.769×(1-0.769), 0.769×(1-0.769)] = [0.178, 0.178]
δ₁ = ∂L/∂z₁ = ∂L/∂h ⊙ ∂h/∂z₁ = [-0.068, -0.057] ⊙ [0.178, 0.178]
            = [-0.012, -0.010]  (⊙ is element-wise multiplication)

Step 5: Hidden layer weight gradients
∂L/∂W₁ = δ₁ × xᵀ = [[-0.012],  × [1, 2] = [[-0.012, -0.024],
                     [-0.010]]              [-0.010, -0.020]]
∂L/∂b₁ = δ₁ = [-0.012, -0.010]
```

### 6.4 General Backpropagation Algorithm

**Algorithm**:
```
1. Forward Pass:
   For each layer l from 1 to L:
     z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
     a⁽ˡ⁾ = σ⁽ˡ⁾(z⁽ˡ⁾)
   Compute loss L

2. Backward Pass:
   Compute output gradient: δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾
   
   For each layer l from L-1 down to 1:
     δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
   
   For each layer l from 1 to L:
     ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
     ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾

3. Update:
   W⁽ˡ⁾ = W⁽ˡ⁾ - α × ∂L/∂W⁽ˡ⁾
   b⁽ˡ⁾ = b⁽ˡ⁾ - α × ∂L/∂b⁽ˡ⁾
```

### 6.5 Backpropagation with Different Activations

**Example 38**: ReLU backward pass
```
Forward: a = ReLU(z) = max(0, z)
Backward: ∂a/∂z = {1 if z > 0; 0 if z ≤ 0}

If z = [-1, 2, 3], a = [0, 2, 3]
Gradient: δ_in = [0.5, -0.3, 0.8]
δ_out = δ_in ⊙ ReLU'(z) = [0.5, -0.3, 0.8] ⊙ [0, 1, 1] = [0, -0.3, 0.8]

Negative pre-activation → zero gradient!
```

**Example 39**: Softmax + Cross-Entropy backward pass

This combination has a beautiful simplification!

```
Forward:
  z = [z₁, z₂, z₃]  (logits)
  a = softmax(z) = [a₁, a₂, a₃]  (probabilities)
  L = -Σᵢ yᵢ log(aᵢ)  (cross-entropy)

Backward (simplified):
  ∂L/∂z = a - y

Example:
  z = [2.0, 1.0, 0.1]
  a = softmax(z) = [0.659, 0.242, 0.099]
  y = [1, 0, 0]  (one-hot encoded true label)
  
  ∂L/∂z = [0.659-1, 0.242-0, 0.099-0] = [-0.341, 0.242, 0.099]
  
Interpretation: Negative gradient for correct class, positive for wrong classes
```

### 6.6 Mini-Batch Backpropagation

**Example 40**: Batch gradient computation
```
Batch of 3 samples:
X = [[1, 2],     y = [1,
     [2, 3],          0,
     [1, 1]]          1]

For each sample i:
  1. Forward pass: compute ŷᵢ
  2. Backward pass: compute ∂Lᵢ/∂W, ∂Lᵢ/∂b

Average gradients:
∂L/∂W = (1/3) Σᵢ ∂Lᵢ/∂W
∂L/∂b = (1/3) Σᵢ ∂Lᵢ/∂b

Single update with averaged gradient
```

### 6.7 Common Issues and Solutions

**Problem 1: Vanishing Gradients**
```
Deep network with sigmoid activations
Layer 10: δ⁽¹⁰⁾ = 0.1
Layer 9: δ⁽⁹⁾ = δ⁽¹⁰⁾ × W⁽¹⁰⁾ × σ'(z⁽⁹⁾) ≈ 0.1 × 0.5 × 0.25 = 0.0125
Layer 8: δ⁽⁸⁾ ≈ 0.0125 × 0.5 × 0.25 = 0.0016
...
Layer 1: δ⁽¹⁾ ≈ 0 (vanished!)

Solution: Use ReLU, batch normalization, residual connections
```

**Problem 2: Exploding Gradients**
```
Weights too large → gradients multiply and explode
δ⁽¹⁾ = δ⁽²⁾ × W⁽²⁾ × σ'(z⁽¹⁾)

If ||W⁽²⁾|| > 10, gradients can grow exponentially

Solution: Gradient clipping, proper weight initialization
```

**Example 41**: Gradient clipping
```
∂L/∂W = [[-5.2, 3.1],
          [8.7, -12.4]]

If ||∂L/∂W|| > threshold (say, 5):
  ∂L/∂W = threshold × (∂L/∂W / ||∂L/∂W||)

||∂L/∂W|| = √(5.2² + 3.1² + 8.7² + 12.4²) = 16.5
∂L/∂W_clipped = 5 × (∂L/∂W / 16.5) ≈ [[-1.58, 0.94], [2.64, -3.76]]
```

---

## Summary

### Key Concepts Recap

**Linear Algebra**: Vectors and matrices represent data and transformations
- Matrix multiplication: Neural network layer computation
- Transpose and inverse: Used in optimization

**Calculus**: Derivatives measure change and enable optimization
- Chain rule: Foundation of backpropagation
- Gradient descent: Iterative parameter updates

**Probability**: Models uncertainty and classification
- Conditional probability: Bayesian inference
- Expected value: Loss function motivation

**Logarithms**: Transform multiplicative to additive relationships
- Numerical stability: Prevents underflow
- Loss functions: Cross-entropy

**Activation Functions**: Introduce non-linearity
- ReLU: Most common for hidden layers
- Softmax: Multi-class output
- Sigmoid: Binary classification output

**Backpropagation**: Efficient gradient computation
- Chain rule application: Propagate errors backward
- Weight updates: Gradient descent with computed gradients

### The Complete Picture

```
1. Initialize network weights randomly
2. For each training iteration:
   a. Forward Pass:
      - Linear: z = Wx + b
      - Activation: a = σ(z)
      - Repeat for each layer
      - Compute loss: L
   
   b. Backward Pass:
      - Compute output gradient
      - Backpropagate through each layer (chain rule)
      - Calculate weight gradients
   
   c. Update:
      - w_new = w_old - α × ∂L/∂w
      - Repeat until convergence
```

This tutorial covered the essential mathematics for understanding and implementing machine learning models, with emphasis on neural networks. Each concept builds on previous ones, culminating in backpropagation—the algorithm that makes deep learning possible.

---

## Practice Problems

1. **Linear Algebra**: Given X (100×5 feature matrix) and W (5×3 weight matrix), what's the shape of XW?

2. **Calculus**: Find the minimum of f(x) = x² - 4x + 7 using gradient descent (2 iterations, α=0.1, start at x=0).

3. **Probability**: P(Disease)=0.01, P(+Test|Disease)=0.95, P(+Test|Healthy)=0.10. Find P(Disease|+Test).

4. **Logarithms**: Why is log(P(A)×P(B)×P(C)) preferred over P(A)×P(B)×P(C) in computation?

5. **Activation**: Compute softmax([1.0, 2.0, 3.0]).

6. **Backpropagation**: For L=(wx-y)², x=3, y=7, w=2, compute ∂L/∂w.

## Further Reading

- **Linear Algebra**: "Introduction to Linear Algebra" by Gilbert Strang
- **Calculus**: MIT OpenCourseWare 18.01 Single Variable Calculus
- **Probability**: "Pattern Recognition and Machine Learning" by Christopher Bishop
- **Deep Learning**: "Deep Learning" by Goodfellow, Bengio, and Courville
- **Implementation**: PyTorch/TensorFlow tutorials for hands-on practice