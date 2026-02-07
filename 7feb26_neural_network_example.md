# Feed-Forward Neural Network: House Price Prediction
## A Complete Numerical Example

## Problem Setup

**Goal:** Predict house prices based on two features:
- Square footage (in hundreds of sq ft)
- Number of bedrooms

**Network Architecture:**
- Input layer: 2 neurons
- Hidden layer: 2 neurons (ReLU activation)
- Output layer: 1 neuron (price in $100k)

**Training Example:**
- Input: Square footage = 2 (200 sq ft), Bedrooms = 3
- Actual price = 3.5 ($350k)

---

## Step 1: Initialize Weights and Biases

Weights and biases are randomly initialized:

```
Hidden Layer Weights:
  w1_1 = 0.5  (square footage → hidden neuron 1)
  w1_2 = 0.3  (bedrooms → hidden neuron 1)
  w2_1 = 0.4  (square footage → hidden neuron 2)
  w2_2 = 0.2  (bedrooms → hidden neuron 2)

Output Layer Weights:
  w3_1 = 0.6  (hidden neuron 1 → output)
  w3_2 = 0.7  (hidden neuron 2 → output)

Biases:
  b1 = 0.1  (hidden neuron 1)
  b2 = 0.1  (hidden neuron 2)
  b3 = 0.1  (output neuron)
```

---

## Step 2: Forward Pass (Make Prediction)

### 2a. Calculate Hidden Layer (Pre-Activation)

**Hidden Neuron 1:**
```
z1 = (input1 × w1_1) + (input2 × w1_2) + b1
z1 = (2 × 0.5) + (3 × 0.3) + 0.1
z1 = 1.0 + 0.9 + 0.1 = 2.0
```

**Hidden Neuron 2:**
```
z2 = (2 × 0.4) + (3 × 0.2) + 0.1
z2 = 0.8 + 0.6 + 0.1 = 1.5
```

### 2b. Apply Activation Function (ReLU)

ReLU(x) = max(0, x) — outputs the value if positive, 0 if negative

```
a1 = ReLU(z1) = ReLU(2.0) = 2.0
a2 = ReLU(z2) = ReLU(1.5) = 1.5
```

### 2c. Calculate Output Layer

```
z3 = (a1 × w3_1) + (a2 × w3_2) + b3
z3 = (2.0 × 0.6) + (1.5 × 0.7) + 0.1
z3 = 1.2 + 1.05 + 0.1 = 2.35
```

**Prediction:** 2.35 ($235k)  
**Actual:** 3.5 ($350k)  
**Error:** -1.15 (we under-predicted)

---

## Step 3: Calculate Loss

Using Mean Squared Error (MSE):

```
Loss = (predicted - actual)²
Loss = (2.35 - 3.5)²
Loss = (-1.15)²
Loss = 1.3225
```

**Goal:** Minimize this loss by adjusting weights.

---

## Step 4: Backpropagation

Calculate how much each weight contributed to the error, working backwards from output to input.

### 4a. Output Layer Gradient

```
∂Loss/∂z3 = 2 × (predicted - actual)
∂Loss/∂z3 = 2 × (2.35 - 3.5)
∂Loss/∂z3 = -2.3
```

### 4b. Output Weight Gradients

```
∂Loss/∂w3_1 = ∂Loss/∂z3 × a1 = -2.3 × 2.0 = -4.6
∂Loss/∂w3_2 = ∂Loss/∂z3 × a2 = -2.3 × 1.5 = -3.45
∂Loss/∂b3 = ∂Loss/∂z3 = -2.3
```

### 4c. Backpropagate to Hidden Layer

```
∂Loss/∂a1 = ∂Loss/∂z3 × w3_1 = -2.3 × 0.6 = -1.38
∂Loss/∂a2 = ∂Loss/∂z3 × w3_2 = -2.3 × 0.7 = -1.61
```

### 4d. Apply ReLU Derivative

ReLU derivative = 1 if input was positive, 0 if negative

Since z1 = 2.0 and z2 = 1.5 were both positive:

```
∂Loss/∂z1 = ∂Loss/∂a1 × 1 = -1.38
∂Loss/∂z2 = ∂Loss/∂a2 × 1 = -1.61
```

### 4e. Hidden Layer Weight Gradients

```
∂Loss/∂w1_1 = ∂Loss/∂z1 × input1 = -1.38 × 2 = -2.76
∂Loss/∂w1_2 = ∂Loss/∂z1 × input2 = -1.38 × 3 = -4.14
∂Loss/∂w2_1 = ∂Loss/∂z2 × input1 = -1.61 × 2 = -3.22
∂Loss/∂w2_2 = ∂Loss/∂z2 × input2 = -1.61 × 3 = -4.83
∂Loss/∂b1 = ∂Loss/∂z1 = -1.38
∂Loss/∂b2 = ∂Loss/∂z2 = -1.61
```

---

## Step 5: Update Weights (Gradient Descent)

Adjust weights in the opposite direction of gradients to reduce loss.

**Learning rate:** α = 0.01 (controls step size)

**Update formula:** weight_new = weight_old - α × gradient

```
w1_1: 0.5 - 0.01 × (-2.76) = 0.5 + 0.0276 = 0.5276
w1_2: 0.3 - 0.01 × (-4.14) = 0.3 + 0.0414 = 0.3414
w2_1: 0.4 - 0.01 × (-3.22) = 0.4 + 0.0322 = 0.4322
w2_2: 0.2 - 0.01 × (-4.83) = 0.2 + 0.0483 = 0.2483
w3_1: 0.6 - 0.01 × (-4.6) = 0.6 + 0.046 = 0.646
w3_2: 0.7 - 0.01 × (-3.45) = 0.7 + 0.0345 = 0.7345
b1: 0.1 - 0.01 × (-1.38) = 0.1138
b2: 0.1 - 0.01 × (-1.61) = 0.1161
b3: 0.1 - 0.01 × (-2.3) = 0.123
```

---

## Step 6: Verify Improvement (Iteration 2)

Using the updated weights, make a new prediction with the same training example.

### Forward Pass with Updated Weights

**Hidden Layer:**
```
z1 = (2 × 0.5276) + (3 × 0.3414) + 0.1138 = 2.1932
z2 = (2 × 0.4322) + (3 × 0.2483) + 0.1161 = 1.7254

a1 = ReLU(2.1932) = 2.1932
a2 = ReLU(1.7254) = 1.7254
```

**Output:**
```
z3 = (2.1932 × 0.646) + (1.7254 × 0.7345) + 0.123 = 2.8071
```

**New Prediction:** 2.8071 ($280.71k)  
**New Loss:** (2.8071 - 3.5)² = 0.4799

**Improvement:** Loss decreased from 1.3225 → 0.4799 (63.7% reduction!)

---

## Step 7: Continue Training

Repeat Steps 2-6 for multiple iterations:

### Iteration Progress

| Iteration | Prediction | Loss | Improvement |
|-----------|------------|------|-------------|
| 1 | $235.00k | 1.3225 | — |
| 2 | $280.71k | 0.4799 | 63.7% ↓ |
| 3 | $334.39k | 0.0244 | 94.9% ↓ |
| 4 | $345.21k | 0.0023 | 90.6% ↓ |
| 5 | $348.73k | 0.0002 | 91.3% ↓ |

**Total improvement:** 99.98% loss reduction in just 5 iterations!

---

## Key Concepts Summary

### Input Layer
The raw features that feed into the network (square footage, bedrooms).

### Weights
Learned parameters that determine feature importance. Higher weights mean stronger influence on the prediction.

### Biases
Adjustable offsets that help the network fit data better.

### Activation Function (ReLU)
Adds non-linearity so the network can learn complex patterns. Without it, the network would only learn linear relationships.

### Forward Pass
Information flows from input → hidden → output to generate a prediction.

### Loss Function
Measures how wrong the prediction is. Lower loss = better model.

### Backpropagation
Calculates gradients (derivatives) showing how much each weight contributed to the error.

### Gradient Descent
Updates weights in the direction that reduces loss:
- Negative gradient → increase weight
- Positive gradient → decrease weight

### Learning Rate
Controls how big each weight update step is:
- Too high: May overshoot optimal values
- Too low: Learns very slowly
- Just right: Steady, consistent improvement

### Training Process
1. Forward pass → make prediction
2. Calculate loss → measure error
3. Backpropagation → find gradients
4. Update weights → improve model
5. Repeat with many examples until loss is minimized

---

## Why It Works

The network starts with random weights (makes poor predictions). Through gradient descent:

1. **Gradients point toward improvement:** They tell us exactly which direction to adjust each weight
2. **Small steps accumulate:** Each iteration makes a small improvement
3. **Loss guides learning:** The network "learns" by minimizing prediction error
4. **Weights encode patterns:** After training, weights represent learned relationships (e.g., "bigger houses cost more")

This is the fundamental mechanism behind all neural networks, from simple models to large language models!