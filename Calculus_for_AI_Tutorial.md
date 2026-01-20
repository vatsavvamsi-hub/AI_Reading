# Calculus for Artificial Intelligence: A Comprehensive Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Derivatives and Differentiation](#derivatives-and-differentiation)
3. [Partial Derivatives](#partial-derivatives)
4. [Gradients and Gradient Descent](#gradients-and-gradient-descent)
5. [Chain Rule and Backpropagation](#chain-rule-and-backpropagation)
6. [Optimization in AI](#optimization-in-ai)
7. [Integration in AI](#integration-in-ai)
8. [Multivariable Calculus](#multivariable-calculus)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

Calculus is the mathematical foundation of modern AI and machine learning. It enables us to:
- Optimize neural network parameters
- Compute gradients for learning algorithms
- Understand how small changes affect model outputs
- Minimize loss functions

This tutorial covers all essential calculus concepts used in AI with practical examples.

---

## Derivatives and Differentiation

### What is a Derivative?

A derivative measures how a function changes as its input changes. For a function f(x), the derivative f'(x) or df/dx represents the rate of change.

**Definition:**
```
f'(x) = lim[h→0] (f(x + h) - f(x)) / h
```

### Basic Derivative Rules

#### Power Rule
```
d/dx [x^n] = n·x^(n-1)
```

**Example:**
```
f(x) = x³
f'(x) = 3x²

f(x) = x⁵
f'(x) = 5x⁴
```

#### Constant Rule
```
d/dx [c] = 0  (where c is a constant)
```

**Example:**
```
f(x) = 7
f'(x) = 0
```

#### Sum Rule
```
d/dx [f(x) + g(x)] = f'(x) + g'(x)
```

**Example:**
```
f(x) = x³ + 2x² + 5
f'(x) = 3x² + 4x + 0 = 3x² + 4x
```

### Common Functions in AI

#### Exponential Function
```
d/dx [e^x] = e^x
```

**Example:**
```
f(x) = e^(2x)
f'(x) = 2e^(2x)  (using chain rule)
```

#### Natural Logarithm
```
d/dx [ln(x)] = 1/x
```

**Example:**
```
f(x) = ln(x²)
f'(x) = (1/x²)·2x = 2/x
```

#### Sigmoid Function (Critical in Neural Networks)
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x)·(1 - σ(x))
```

**Example Derivation:**
```
Let σ(x) = (1 + e^(-x))^(-1)
Using chain rule:
σ'(x) = -1·(1 + e^(-x))^(-2)·(-e^(-x))
      = e^(-x) / (1 + e^(-x))²
      = σ(x)·(1 - σ(x))
```

**Numerical Example:**
```
x = 0
σ(0) = 1/(1 + e^0) = 1/2 = 0.5
σ'(0) = 0.5·(1 - 0.5) = 0.25
```

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

**Example:**
```
x = -2: ReLU(-2) = 0, ReLU'(-2) = 0
x = 3:  ReLU(3) = 3,  ReLU'(3) = 1
```

#### Tanh (Hyperbolic Tangent)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

**Example:**
```
x = 0
tanh(0) = 0
tanh'(0) = 1 - 0² = 1
```

---

## Partial Derivatives

When functions have multiple variables, we use partial derivatives to measure the rate of change with respect to one variable while keeping others constant.

### Notation
```
∂f/∂x  or  f_x  or  ∂_x f
```

### Definition
```
∂f/∂x = lim[h→0] (f(x + h, y) - f(x, y)) / h
```

### Examples

**Example 1: Simple Function**
```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y  (treat y as constant)
∂f/∂y = 3x + 2y  (treat x as constant)

At point (2, 1):
∂f/∂x = 2(2) + 3(1) = 7
∂f/∂y = 3(2) + 2(1) = 8
```

**Example 2: Neural Network Weight**
```
f(w₁, w₂) = (w₁·x₁ + w₂·x₂ - y)²

where x₁ = 2, x₂ = 3, y = 5

∂f/∂w₁ = 2(w₁·x₁ + w₂·x₂ - y)·x₁
       = 2(2w₁ + 3w₂ - 5)·2
       = 4(2w₁ + 3w₂ - 5)

∂f/∂w₂ = 2(w₁·x₁ + w₂·x₂ - y)·x₂
       = 2(2w₁ + 3w₂ - 5)·3
       = 6(2w₁ + 3w₂ - 5)

At w₁ = 1, w₂ = 1:
∂f/∂w₁ = 4(2 + 3 - 5) = 0
∂f/∂w₂ = 6(2 + 3 - 5) = 0
```

**Example 3: Mean Squared Error**
```
L(w, b) = (1/n)·Σᵢ(wxᵢ + b - yᵢ)²

∂L/∂w = (2/n)·Σᵢ(wxᵢ + b - yᵢ)·xᵢ
∂L/∂b = (2/n)·Σᵢ(wxᵢ + b - yᵢ)

For n = 2, x = [1, 2], y = [2, 4], w = 1.5, b = 0.5:
Predictions: [2.0, 3.5]
Errors: [0, -0.5]

∂L/∂w = (2/2)·(0·1 + (-0.5)·2) = -1.0
∂L/∂b = (2/2)·(0 + (-0.5)) = -0.5
```

---

## Gradients and Gradient Descent

### The Gradient Vector

The gradient is a vector of all partial derivatives. It points in the direction of steepest ascent.

**Definition:**
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

### Example: 2D Gradient
```
f(x, y) = x² + y²

∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

At point (3, 4):
∇f(3, 4) = [6, 8]
```

### Gradient Descent Algorithm

Gradient descent minimizes a function by moving in the direction opposite to the gradient.

**Update Rule:**
```
θₙₑw = θₒₗd - α·∇f(θₒₗd)
```
where α is the learning rate.

### Example: Minimizing a Quadratic Function

**Problem:** Minimize f(x) = x² - 4x + 5

```
f'(x) = 2x - 4

Starting point: x₀ = 0
Learning rate: α = 0.1

Iteration 1:
x₁ = x₀ - α·f'(x₀)
   = 0 - 0.1·(2·0 - 4)
   = 0 - 0.1·(-4)
   = 0.4

Iteration 2:
x₂ = x₁ - α·f'(x₁)
   = 0.4 - 0.1·(2·0.4 - 4)
   = 0.4 - 0.1·(-3.2)
   = 0.72

Iteration 3:
x₃ = 0.72 - 0.1·(2·0.72 - 4)
   = 0.72 - 0.1·(-2.56)
   = 0.976

...converges to x = 2 (the minimum)
```

### Example: Linear Regression with Gradient Descent

**Problem:** Fit y = wx + b to data points: (1, 3), (2, 5), (3, 7)

```
Loss: L(w, b) = (1/3)·Σᵢ(wxᵢ + b - yᵢ)²

Gradients:
∂L/∂w = (2/3)·Σᵢ(wxᵢ + b - yᵢ)·xᵢ
∂L/∂b = (2/3)·Σᵢ(wxᵢ + b - yᵢ)

Initial: w₀ = 0, b₀ = 0, α = 0.01

Iteration 1:
Predictions: [0, 0, 0]
Errors: [-3, -5, -7]

∂L/∂w = (2/3)·((-3)·1 + (-5)·2 + (-7)·3) = (2/3)·(-34) = -22.67
∂L/∂b = (2/3)·(-3 - 5 - 7) = -10

w₁ = 0 - 0.01·(-22.67) = 0.227
b₁ = 0 - 0.01·(-10) = 0.1

Continue iterations until convergence...
True solution: w = 2, b = 1
```

---

## Chain Rule and Backpropagation

The chain rule is the foundation of backpropagation in neural networks.

### Single Variable Chain Rule
```
If y = f(g(x)), then dy/dx = f'(g(x))·g'(x)
```

**Example:**
```
y = (x² + 1)³

Let u = x² + 1, then y = u³

dy/du = 3u²
du/dx = 2x

dy/dx = dy/du · du/dx
      = 3u² · 2x
      = 3(x² + 1)² · 2x
      = 6x(x² + 1)²

At x = 2:
dy/dx = 6·2·(4 + 1)² = 12·25 = 300
```

### Multivariable Chain Rule
```
If z = f(x, y) where x = g(t) and y = h(t), then:
dz/dt = (∂f/∂x)·(dx/dt) + (∂f/∂y)·(dy/dt)
```

**Example:**
```
z = x² + y²
x = cos(t)
y = sin(t)

∂z/∂x = 2x
∂z/∂y = 2y
dx/dt = -sin(t)
dy/dt = cos(t)

dz/dt = 2x·(-sin(t)) + 2y·cos(t)
      = 2cos(t)·(-sin(t)) + 2sin(t)·cos(t)
      = 0
```

### Backpropagation Example: Simple Neural Network

**Architecture:** Input → Hidden → Output

```
Network:
x → h = σ(w₁·x + b₁) → y = σ(w₂·h + b₂)

Loss: L = (y - target)²

Forward pass (x = 0.5, target = 0.8, w₁ = 0.4, w₂ = 0.6, b₁ = 0.1, b₂ = 0.2):

Step 1: z₁ = w₁·x + b₁ = 0.4·0.5 + 0.1 = 0.3
Step 2: h = σ(z₁) = 1/(1 + e^(-0.3)) ≈ 0.574
Step 3: z₂ = w₂·h + b₂ = 0.6·0.574 + 0.2 ≈ 0.545
Step 4: y = σ(z₂) = 1/(1 + e^(-0.545)) ≈ 0.633
Step 5: L = (0.633 - 0.8)² ≈ 0.028

Backward pass (using chain rule):

∂L/∂y = 2(y - target) = 2(0.633 - 0.8) = -0.334

∂L/∂z₂ = ∂L/∂y · ∂y/∂z₂
        = -0.334 · σ(z₂)·(1 - σ(z₂))
        = -0.334 · 0.633·(1 - 0.633)
        ≈ -0.078

∂L/∂w₂ = ∂L/∂z₂ · ∂z₂/∂w₂
        = -0.078 · h
        = -0.078 · 0.574
        ≈ -0.045

∂L/∂b₂ = ∂L/∂z₂ · ∂z₂/∂b₂
        = -0.078 · 1
        = -0.078

∂L/∂h = ∂L/∂z₂ · ∂z₂/∂h
       = -0.078 · w₂
       = -0.078 · 0.6
       ≈ -0.047

∂L/∂z₁ = ∂L/∂h · ∂h/∂z₁
        = -0.047 · σ(z₁)·(1 - σ(z₁))
        = -0.047 · 0.574·(1 - 0.574)
        ≈ -0.011

∂L/∂w₁ = ∂L/∂z₁ · ∂z₁/∂w₁
        = -0.011 · x
        = -0.011 · 0.5
        ≈ -0.006

∂L/∂b₁ = ∂L/∂z₁ · ∂z₁/∂b₁
        = -0.011 · 1
        = -0.011

Update (α = 0.1):
w₂ := 0.6 - 0.1·(-0.045) = 0.6045
b₂ := 0.2 - 0.1·(-0.078) = 0.2078
w₁ := 0.4 - 0.1·(-0.006) = 0.4006
b₁ := 0.1 - 0.1·(-0.011) = 0.1011
```

---

## Optimization in AI

### Critical Points and Extrema

A critical point occurs where the gradient is zero: ∇f = 0

**Types:**
- Minimum: ∇f = 0 and f curves upward
- Maximum: ∇f = 0 and f curves downward
- Saddle point: ∇f = 0 but neither min nor max

### Second Derivative Test (1D)
```
If f'(x₀) = 0:
- f''(x₀) > 0 → local minimum
- f''(x₀) < 0 → local maximum
- f''(x₀) = 0 → inconclusive
```

**Example:**
```
f(x) = x³ - 3x² + 2

f'(x) = 3x² - 6x = 3x(x - 2)
Critical points: x = 0, x = 2

f''(x) = 6x - 6

At x = 0: f''(0) = -6 < 0 → local maximum
At x = 2: f''(2) = 6 > 0 → local minimum
```

### Hessian Matrix (Multivariable)

The Hessian is a matrix of second-order partial derivatives.

```
H = [∂²f/∂x₁²    ∂²f/∂x₁∂x₂]
    [∂²f/∂x₂∂x₁  ∂²f/∂x₂²  ]
```

**Example:**
```
f(x, y) = x² + xy + y²

∂f/∂x = 2x + y
∂f/∂y = x + 2y

Critical point (∇f = 0):
2x + y = 0
x + 2y = 0
Solution: x = 0, y = 0

Hessian:
∂²f/∂x² = 2
∂²f/∂y² = 2
∂²f/∂x∂y = 1

H = [2  1]
    [1  2]

Eigenvalues: λ₁ = 3, λ₂ = 1 (both positive)
Therefore, (0, 0) is a local minimum.
```

### Convex Functions in ML

A function is convex if its second derivative (or Hessian) is always non-negative.

**Example: Convex Loss Functions**
```
1. Mean Squared Error:
   L(w) = (1/n)·Σᵢ(wᵀxᵢ - yᵢ)²
   Convex: Yes (quadratic in w)

2. Logistic Loss:
   L(w) = Σᵢ log(1 + exp(-yᵢ·wᵀxᵢ))
   Convex: Yes

3. Hinge Loss (SVM):
   L(w) = Σᵢ max(0, 1 - yᵢ·wᵀxᵢ)
   Convex: Yes
```

### Optimization Algorithms

#### Stochastic Gradient Descent (SGD)
```
For each mini-batch:
  θ := θ - α·∇L(θ; batch)
```

#### SGD with Momentum
```
v := β·v + ∇L(θ)
θ := θ - α·v
```

**Example:**
```
Minimize f(x) = x²
f'(x) = 2x

x₀ = 10, v₀ = 0, α = 0.1, β = 0.9

Iteration 1:
v₁ = 0.9·0 + 2·10 = 20
x₁ = 10 - 0.1·20 = 8

Iteration 2:
v₂ = 0.9·20 + 2·8 = 34
x₂ = 8 - 0.1·34 = 4.6

Iteration 3:
v₃ = 0.9·34 + 2·4.6 = 39.8
x₃ = 4.6 - 0.1·39.8 = 0.62

(Converges faster than standard GD)
```

#### Adam Optimizer
```
m := β₁·m + (1 - β₁)·∇L(θ)        (first moment)
v := β₂·v + (1 - β₂)·(∇L(θ))²    (second moment)
m̂ := m/(1 - β₁ᵗ)                 (bias correction)
v̂ := v/(1 - β₂ᵗ)
θ := θ - α·m̂/√(v̂ + ε)
```

---

## Integration in AI

### Definite Integrals

**Definition:**
```
∫ₐᵇ f(x)dx = lim[n→∞] Σᵢ f(xᵢ)·Δx
```

### Fundamental Theorem of Calculus
```
∫ₐᵇ f(x)dx = F(b) - F(a)
where F'(x) = f(x)
```

**Example:**
```
∫₀² x²dx = [x³/3]₀² = 8/3 - 0 = 8/3 ≈ 2.667
```

### Applications in AI

#### Probability Distributions
```
For probability density function p(x):
∫₋∞^∞ p(x)dx = 1

Example: Gaussian Distribution
p(x) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))

∫₋∞^∞ p(x)dx = 1 ✓
```

#### Expected Value
```
E[X] = ∫₋∞^∞ x·p(x)dx
```

**Example:**
```
p(x) = 1 for x ∈ [0, 1] (uniform distribution)

E[X] = ∫₀¹ x·1·dx = [x²/2]₀¹ = 1/2
```

#### Variance
```
Var(X) = E[X²] - (E[X])²
       = ∫₋∞^∞ x²·p(x)dx - (∫₋∞^∞ x·p(x)dx)²
```

**Example:**
```
p(x) = 1 for x ∈ [0, 1]

E[X²] = ∫₀¹ x²·dx = [x³/3]₀¹ = 1/3
E[X] = 1/2

Var(X) = 1/3 - (1/2)² = 1/3 - 1/4 = 1/12
```

#### Evidence Lower Bound (ELBO) in Variational Inference
```
ELBO = ∫ q(z)·log(p(x,z)/q(z))dz
     = E_q[log p(x,z)] - E_q[log q(z)]
```

### Monte Carlo Integration

Used when integrals are intractable.

**Formula:**
```
∫ₐᵇ f(x)dx ≈ ((b-a)/n)·Σᵢ f(xᵢ)
where xᵢ are random samples from [a, b]
```

**Example:**
```
Estimate ∫₀¹ x²dx using 4 random samples

Samples: x₁ = 0.2, x₂ = 0.5, x₃ = 0.7, x₄ = 0.9

f(x₁) = 0.04
f(x₂) = 0.25
f(x₃) = 0.49
f(x₄) = 0.81

∫₀¹ x²dx ≈ (1/4)·(0.04 + 0.25 + 0.49 + 0.81)
         = 0.3975

True value: 1/3 ≈ 0.333
(Error decreases with more samples)
```

---

## Multivariable Calculus

### Jacobian Matrix

The Jacobian contains all first-order partial derivatives of a vector-valued function.

**Definition:**
```
For f: ℝⁿ → ℝᵐ
     
J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
    [   ⋮        ⋮      ⋱      ⋮   ]
    [∂fᵐ/∂x₁  ∂fᵐ/∂x₂  ...  ∂fᵐ/∂xₙ]
```

**Example: Neural Network Layer**
```
f(x₁, x₂) = [f₁, f₂]
where:
f₁ = x₁² + x₂
f₂ = x₁·x₂

Jacobian:
J = [∂f₁/∂x₁  ∂f₁/∂x₂]   [2x₁   1 ]
    [∂f₂/∂x₁  ∂f₂/∂x₂] = [x₂   x₁]

At point (2, 3):
J(2,3) = [4  1]
         [3  2]
```

**Example: Batch Normalization Gradient**
```
For batch normalization:
y = (x - μ)/σ

where μ = mean(x), σ = std(x)

The Jacobian captures how each output depends on all inputs.
```

### Directional Derivatives

The rate of change of f in direction v.

**Formula:**
```
D_v f = ∇f · v̂
where v̂ is the unit vector in direction v
```

**Example:**
```
f(x, y) = x² + y²
∇f = [2x, 2y]

At point (3, 4), in direction v = [1, 1]:

v̂ = v/||v|| = [1/√2, 1/√2]

D_v f(3,4) = [6, 8] · [1/√2, 1/√2]
           = (6 + 8)/√2
           = 14/√2
           ≈ 9.90
```

### Lagrange Multipliers (Constrained Optimization)

Used to optimize f(x, y) subject to constraint g(x, y) = 0.

**Method:**
```
∇f = λ·∇g
g(x, y) = 0
```

**Example:**
```
Maximize f(x, y) = xy
Subject to: x + y = 10

g(x, y) = x + y - 10 = 0

∇f = [y, x]
∇g = [1, 1]

System:
y = λ·1  →  y = λ
x = λ·1  →  x = λ
x + y = 10

From first two: x = y
From third: 2x = 10  →  x = 5
Therefore: x = 5, y = 5

Maximum value: f(5, 5) = 25
```

**AI Application: Support Vector Machines**
```
Minimize: (1/2)||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i

Using Lagrange multipliers leads to the dual formulation.
```

### Taylor Series Expansion

Approximates functions using derivatives.

**Formula:**
```
f(x) ≈ f(a) + f'(a)(x-a) + (f''(a)/2!)(x-a)² + ...
```

**Example:**
```
Approximate e^x near x = 0

f(x) = e^x
f'(x) = e^x
f''(x) = e^x
...

At a = 0:
e^x ≈ 1 + x + x²/2 + x³/6 + ...

For x = 0.1:
e^0.1 ≈ 1 + 0.1 + 0.01/2 + 0.001/6
      ≈ 1.10517

True value: 1.10517 ✓
```

**Multivariable Taylor Series:**
```
f(x, y) ≈ f(a, b) + ∂f/∂x·(x-a) + ∂f/∂y·(y-b)
         + (1/2)[∂²f/∂x²·(x-a)² + 2·∂²f/∂x∂y·(x-a)(y-b) + ∂²f/∂y²·(y-b)²]
```

---

## Advanced Topics

### Automatic Differentiation

Modern deep learning frameworks use automatic differentiation (autodiff).

**Forward Mode Example:**
```
Compute f(x) = x·sin(x) and f'(x) at x = π

Dual numbers: (value, derivative)

x = (π, 1)
sin(x) = (sin(π), cos(π)·1) = (0, -1)
x·sin(x) = (π·0, π·(-1) + 0·1) = (0, -π)

Result: f(π) = 0, f'(π) = -π
```

**Reverse Mode (Used in Backpropagation):**
```
Forward pass: compute values
Backward pass: compute gradients

Example: f(x, y) = (x + y)·(x·y)

Forward:
a = x + y
b = x·y
c = a·b

Backward (∂f/∂x):
∂c/∂a = b,  ∂c/∂b = a
∂a/∂x = 1,  ∂b/∂x = y

∂f/∂x = ∂c/∂a·∂a/∂x + ∂c/∂b·∂b/∂x
      = b·1 + a·y
      = xy + (x+y)·y
      = xy + xy + y²
      = 2xy + y²
```

### Vector Calculus in Deep Learning

#### Gradient of Vector-Matrix Operations
```
For y = Wx (where W is m×n, x is n×1):

∂y/∂x = Wᵀ  (Jacobian)

If L = ||y||², then:
∂L/∂x = 2Wᵀy
```

**Example:**
```
W = [1  2]    x = [3]
    [3  4]        [4]

y = Wx = [11]
         [25]

If L = y₁² + y₂² = 121 + 625 = 746

∂L/∂x = 2Wᵀy = 2·[1  3][11]  = 2·[172]  = [344]
                  [2  4][25]      [242]    [484]
```

#### Hadamard Product (Element-wise)
```
If z = x ⊙ y (element-wise product)
∂z/∂x = diag(y)
```

### Calculus in Activation Functions

#### Softmax Function
```
softmax(xᵢ) = e^xᵢ / Σⱼ e^xʲ

Jacobian:
∂softmax(xᵢ)/∂xⱼ = {
  softmax(xᵢ)·(1 - softmax(xⱼ))  if i = j
  -softmax(xᵢ)·softmax(xⱼ)        if i ≠ j
}
```

**Example:**
```
x = [1, 2, 3]

e^x = [2.718, 7.389, 20.086]
Sum = 30.193

softmax(x) = [0.090, 0.245, 0.665]

∂softmax(x₁)/∂x₁ = 0.090·(1 - 0.090) = 0.082
∂softmax(x₁)/∂x₂ = -0.090·0.245 = -0.022
```

#### Leaky ReLU
```
LeakyReLU(x) = max(αx, x) where α = 0.01

LeakyReLU'(x) = {
  1    if x > 0
  α    if x ≤ 0
}
```

### Cross-Entropy Loss Gradient

**Binary Cross-Entropy:**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)

If ŷ = σ(z) (sigmoid):
∂L/∂z = ŷ - y
```

**Example:**
```
y = 1 (true label)
z = 0.5
ŷ = σ(0.5) = 0.622

∂L/∂z = 0.622 - 1 = -0.378

Update with α = 0.1:
z_new = 0.5 - 0.1·(-0.378) = 0.538
```

**Categorical Cross-Entropy:**
```
L = -Σᵢ yᵢ·log(ŷᵢ)

where ŷ = softmax(z)

∂L/∂zᵢ = ŷᵢ - yᵢ
```

### Calculus in Convolutional Neural Networks

**Convolution Operation:**
```
(f ∗ g)(x) = ∫ f(τ)·g(x - τ)dτ

Discrete:
(f ∗ g)[n] = Σₘ f[m]·g[n - m]
```

**Gradient of Convolution:**
```
If y = x ∗ w, then:
∂L/∂x = ∂L/∂y ∗ flip(w)
∂L/∂w = x ∗ ∂L/∂y
```

**Example:**
```
Input: x = [1, 2, 3, 4]
Kernel: w = [1, 0]
Output: y = [1, 2, 3, 4]

If ∂L/∂y = [1, 1, 1, 1]:
∂L/∂w = Σᵢ xᵢ·(∂L/∂y)ᵢ = [10, 9]
```

### Calculus in Recurrent Neural Networks

**RNN Forward Pass:**
```
hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)
```

**Backpropagation Through Time (BPTT):**
```
∂L/∂hₜ = ∂L/∂hₜ₊₁ · ∂hₜ₊₁/∂hₜ + ∂L/∂yₜ · ∂yₜ/∂hₜ

Involves chain rule across time steps.
```

**Vanishing Gradient Example:**
```
∂hₜ/∂h₀ = ∏ₖ₌₁ᵗ ∂hₖ/∂hₖ₋₁

If |∂hₖ/∂hₖ₋₁| < 1 (typically 0.25-0.5):
After 10 steps: (0.5)¹⁰ ≈ 0.001 → vanishing gradient
```

### Information Theory and Calculus

**Kullback-Leibler Divergence:**
```
KL(P||Q) = ∫ p(x)·log(p(x)/q(x))dx

∂KL/∂θ where q(x) = q(x; θ)

Used in Variational Autoencoders (VAEs)
```

### Calculus of Regularization

**L2 Regularization:**
```
L_total = L_data + λ·||w||²

∂L_total/∂w = ∂L_data/∂w + 2λw
```

**Example:**
```
L_data = (wx - y)² where x = 2, y = 5, w = 3, λ = 0.1

∂L_data/∂w = 2(wx - y)·x = 2(6 - 5)·2 = 4
∂(λ||w||²)/∂w = 2λw = 2·0.1·3 = 0.6

∂L_total/∂w = 4 + 0.6 = 4.6
```

---

## Practice Problems

### Problem 1: Gradient Descent
```
Minimize f(x, y) = x² + 4y² starting from (4, 2) with α = 0.1

Solution:
∇f = [2x, 8y]

Iteration 0: (4, 2)
∇f = [8, 16]
(x₁, y₁) = (4, 2) - 0.1·[8, 16] = (3.2, 0.4)

Iteration 1: (3.2, 0.4)
∇f = [6.4, 3.2]
(x₂, y₂) = (3.2, 0.4) - 0.1·[6.4, 3.2] = (2.56, 0.08)

...continues to (0, 0)
```

### Problem 2: Backpropagation
```
Network: x → z = w·x → a = σ(z) → L = (a - y)²

Given: x = 1, y = 1, w = 0.5

Forward:
z = 0.5·1 = 0.5
a = σ(0.5) ≈ 0.622
L = (0.622 - 1)² ≈ 0.143

Backward:
∂L/∂a = 2(a - y) = 2(0.622 - 1) = -0.756
∂a/∂z = a(1 - a) = 0.622·0.378 ≈ 0.235
∂z/∂w = x = 1

∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
      = -0.756 · 0.235 · 1
      ≈ -0.178
```

### Problem 3: Batch Normalization Derivative
```
Given: x = [1, 3, 5]
Find: ∂(x - mean(x))/∂x₁

μ = (1 + 3 + 5)/3 = 3
y = x - μ = [-2, 0, 2]

∂y₁/∂x₁ = ∂/∂x₁(x₁ - (x₁ + x₂ + x₃)/3)
        = 1 - 1/3
        = 2/3

∂y₂/∂x₁ = ∂/∂x₁(x₂ - (x₁ + x₂ + x₃)/3)
        = -1/3
```

---

## Summary

This tutorial covered:
1. **Derivatives**: Power rule, chain rule, activation functions
2. **Partial Derivatives**: Multivariable calculus fundamentals
3. **Gradients**: Gradient descent and optimization
4. **Backpropagation**: Chain rule in neural networks
5. **Advanced Topics**: Jacobians, Hessians, autodiff
6. **Applications**: Loss functions, regularization, CNNs, RNNs

**Key Takeaways:**
- Calculus enables learning in AI through gradient-based optimization
- The chain rule is the foundation of backpropagation
- Understanding derivatives helps debug and improve models
- Vector calculus extends these concepts to high-dimensional spaces

**Further Study:**
- Convex optimization
- Stochastic processes in ML
- Differential geometry in deep learning
- Calculus of variations

---

**References:**
- Deep Learning (Goodfellow, Bengio, Courville)
- Pattern Recognition and Machine Learning (Bishop)
- Convex Optimization (Boyd & Vandenberghe)
- Mathematics for Machine Learning (Deisenroth et al.)
