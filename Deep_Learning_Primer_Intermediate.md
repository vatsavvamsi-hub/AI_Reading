# Deep Learning Primer: Intermediate Concepts

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Activation Functions](#activation-functions)
4. [Loss Functions and Optimization](#loss-functions-and-optimization)
5. [Convolutional Neural Networks](#convolutional-neural-networks)
6. [Recurrent Neural Networks](#recurrent-neural-networks)
7. [Training Techniques](#training-techniques)
8. [Regularization Methods](#regularization-methods)
9. [Advanced Architectures](#advanced-architectures)

---

## Fundamentals

### What is Deep Learning?
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to learn hierarchical representations of data. It excels at feature extraction from raw inputs without manual feature engineering.

### Biological Motivation
Neural networks are inspired by biological neurons:
- **Neuron**: Receives inputs, applies weights, adds bias, and passes through activation function
- **Synapse**: Represented by weights that are learned during training
- **Learning**: Synaptic strength adjusts based on experience (similar to backpropagation)

### Mathematical Foundation
A single neuron computes:
```
output = activation(Σ(weight_i × input_i) + bias)
```

The network's job is to learn optimal weights and biases that minimize prediction error.

---

## Neural Network Architecture

### Layers and Structure

**Input Layer**: Receives raw data (images, text, time-series, etc.)

**Hidden Layers**: Perform feature transformation and abstraction
- Early layers learn low-level features (edges, textures in images)
- Later layers learn high-level features (objects, concepts)

**Output Layer**: Produces final predictions
- For classification: softmax activation with one neuron per class
- For regression: linear activation with single or multiple neurons

### Depth vs Width
- **Deep networks** (many layers): Better at learning hierarchical features, harder to train
- **Wide networks** (many neurons per layer): Better at memorizing, faster computation per layer

### Parameter Count
Total parameters = Σ(input_size × output_size + output_size) for each layer
- More parameters → more expressiveness but higher risk of overfitting
- Computational cost scales roughly with total parameters

---

## Activation Functions

Activation functions introduce non-linearity, enabling networks to learn complex patterns. Without them, stacking layers would be equivalent to a single linear transformation.

### Common Activation Functions

**ReLU (Rectified Linear Unit)**
```
f(x) = max(0, x)
```
- Most popular for hidden layers
- Fast computation, sparse activations
- Suffers from "dying ReLU" problem (neurons output 0 and stop learning)

**Leaky ReLU**
```
f(x) = x if x > 0, else αx (where α ≈ 0.01)
```
- Addresses dying ReLU problem
- Allows small negative values through

**Sigmoid**
```
f(x) = 1 / (1 + e^(-x))
```
- Output range: (0, 1)
- Historically popular, now mainly used in output layers for binary classification
- Suffers from vanishing gradients in deep networks

**Tanh (Hyperbolic Tangent)**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Output range: (-1, 1)
- Zero-centered, often better than sigmoid
- Still suffers from vanishing gradients

**Softmax** (for multi-class classification)
```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```
- Converts logits to probability distribution
- Output sums to 1
- Used in output layer for classification

---

## Loss Functions and Optimization

### Loss Functions

**Mean Squared Error (MSE)**
```
L = (1/n) × Σ(y_true - y_pred)²
```
- Used for regression problems
- Penalizes large errors heavily (quadratic penalty)

**Cross-Entropy Loss**
```
L = -Σ(y_true × log(y_pred))
```
- Standard for classification
- Measures divergence between predicted and true probability distributions
- Binary cross-entropy for binary classification, categorical for multi-class

**Mean Absolute Error (MAE)**
```
L = (1/n) × Σ|y_true - y_pred|
```
- More robust to outliers than MSE
- Less commonly used in deep learning

### Optimization Algorithms

**Stochastic Gradient Descent (SGD)**
```
weight_new = weight_old - learning_rate × gradient
```
- Update weights using gradient from small batches of data
- Simple, interpretable, but can oscillate and converge slowly
- Momentum variant helps with oscillations

**Adam (Adaptive Moment Estimation)**
```
Combines momentum and adaptive learning rates per parameter
```
- Most popular optimizer in modern deep learning
- Maintains exponential moving averages of gradients and squared gradients
- Generally requires less hyperparameter tuning than SGD

**RMSprop**
```
Divides learning rate by exponential moving average of squared gradients
```
- Adaptive learning rates per parameter
- Often performs well with RNNs

**Learning Rate**
- Too high: Training diverges or oscillates
- Too low: Training is slow and may get stuck in local minima
- Learning rate schedules adjust rate during training (decay, warm-up, etc.)

---

## Convolutional Neural Networks

### Motivation
CNNs exploit spatial structure in data (especially images) through:
- **Local connectivity**: Neurons connect to small spatial regions
- **Weight sharing**: Same filters applied across entire input
- **Hierarchical feature extraction**: Progressively larger receptive fields

### Convolution Operation
A filter (kernel) slides over input, computing dot products at each position:
```
output[i,j] = Σ Σ filter[m,n] × input[i+m, j+n]
```

### Key Concepts

**Filters/Kernels**: Learned weights that detect features (edges, textures, patterns)

**Receptive Field**: The region of input that influences a neuron's output
- Early layers: Small receptive fields (local features)
- Deep layers: Large receptive fields (global features)

**Stride**: How many pixels the filter moves between applications
- Stride=1: Dense output (more computation)
- Stride>1: Reduces spatial dimensions (downsampling)

**Padding**: Adding zeros around input edges
- "Same" padding: Output size = input size
- "Valid" padding: No padding, output size < input size

**Pooling Layers**: Downsample feature maps
- Max pooling: Takes maximum value in window
- Average pooling: Takes average value
- Reduces parameters, computation, and helps with invariance

### Typical CNN Architecture
```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → ... → Flatten → Dense → Output
```

### Famous Architectures
- **LeNet**: Pioneering CNN for digit recognition
- **AlexNet**: Breakthrough on ImageNet (2012)
- **VGG**: Showed that very deep networks work well
- **ResNet**: Residual connections enable very deep networks (100+ layers)
- **Inception**: Multi-scale feature extraction

---

## Recurrent Neural Networks

### Motivation
RNNs process sequential data by maintaining hidden state that captures context:
- Sequence modeling (language, music, time-series)
- Variable-length inputs
- Temporal dependencies

### Basic RNN
```
h_t = activation(W_h × h_{t-1} + W_x × x_t + b)
y_t = W_y × h_t + b_y
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `W_h`, `W_x`, `W_y`: Learnable weight matrices
- Same weights used at every time step (parameter sharing)

### Vanishing/Exploding Gradients
**Problem**: Gradients can exponentially shrink or grow through time steps, making learning difficult
- Vanishing: Early time steps barely influence learning
- Exploding: Weights update too aggressively, training becomes unstable

**Solutions**:
- Gradient clipping: Cap gradient magnitude
- LSTM and GRU architectures

### Long Short-Term Memory (LSTM)

LSTM cells address vanishing gradients with gates that control information flow:

**Forget Gate**: Controls what information to discard
```
f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)
```

**Input Gate**: Controls what new information to store
```
i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)
```

**Cell State Update**: Long-term memory
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Output Gate**: Controls what to output
```
o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

LSTM cells allow gradients to flow effectively through many time steps.

### Gated Recurrent Unit (GRU)
Simplified LSTM with fewer gates:
- Combines forget and input gates
- Fewer parameters, faster training
- Often performs similarly to LSTM

### Bidirectional RNNs
Process sequence forward and backward, concatenating hidden states
- Better for tasks where future context is important (NLP)
- Cannot be used for real-time prediction (requires entire sequence)

---

## Training Techniques

### Backpropagation
Core algorithm for computing gradients:
1. Forward pass: Compute predictions and loss
2. Backward pass: Compute gradients via chain rule
3. Update: Adjust weights using gradients and optimizer

### Batch Training
- **Full batch**: Use entire dataset (slow, stable gradients)
- **Mini-batch**: Use subset of data (standard practice, balanced trade-off)
- **Stochastic**: Use single sample (noisy, helps regularization)

**Batch size trade-offs**:
- Larger batches: More stable gradients, less regularization, faster per-epoch
- Smaller batches: More noisy gradients, better regularization, slower per-epoch

### Epochs and Iterations
- **Epoch**: One complete pass through training data
- **Iteration**: One weight update (usually on a mini-batch)
- Iterations per epoch = dataset_size / batch_size

### Early Stopping
Monitor validation loss during training:
- Stop when validation loss stops improving
- Prevents overfitting without manual tuning
- Requires separate validation set (different from test set)

### Data Augmentation
Create synthetic variations of training data:
- Images: Rotation, flipping, cropping, brightness/contrast changes
- Text: Paraphrasing, back-translation, synonyms
- Audio: Pitch shifting, time stretching, adding noise

Benefits:
- Increases effective dataset size
- Improves generalization
- Provides invariance to transformations

---

## Regularization Methods

Techniques to prevent overfitting and improve generalization.

### L1 and L2 Regularization
Add penalty term to loss function:
```
L_regularized = L + λ × (L1_or_L2_norm_of_weights)
```

**L2 (weight decay)**:
- Penalizes large weights
- Encourages smooth functions
- Most common

**L1**:
- Encourages sparsity (many weights → 0)
- Useful for feature selection

### Dropout
Randomly drop (set to zero) fraction p of neurons during training:
- Forces network to learn redundant representations
- Acts as ensemble of sub-networks
- Typically p=0.5 for hidden layers
- Must scale activations during inference (or use inverted dropout)

Benefits:
- Simple, effective regularization
- Reduces co-adaptation of neurons
- Improves generalization

### Batch Normalization
Normalize layer inputs to have zero mean, unit variance:
```
x_normalized = (x - batch_mean) / √(batch_variance + ε)
y = γ × x_normalized + β
```

Benefits:
- Stabilizes training, allows higher learning rates
- Reduces internal covariate shift
- Acts as weak regularizer
- Can replace dropout in some cases

Considerations:
- Behavior differs between training and inference (uses running statistics at test time)
- Small batch sizes can hurt performance

### Layer Normalization
Similar to batch norm but normalizes across features instead of batch:
- More stable with small batch sizes
- Better for RNNs and Transformers
- Does not require running statistics

---

## Advanced Architectures

### Residual Networks (ResNet)
Solves training degradation in very deep networks via skip connections:
```
output = activation(Conv(x) + x)  # Skip connection
```

Benefits:
- Enables training networks with 100+ layers
- Gradients can flow directly through skip connections
- Reduces vanishing gradient problem

### Attention Mechanism
Learn which parts of input to focus on:
```
attention_weights = softmax(queries × keys^T / √d_k)
output = attention_weights × values
```

Components:
- **Query**: What to look for
- **Key**: What information is available
- **Value**: Actual information
- Often computed multiple times in parallel (multi-head attention)

Benefits:
- Dynamic focus on relevant parts
- Better for long-range dependencies
- Interpretable: attention weights show focus

### Transformers
Stacks of multi-head self-attention with feed-forward networks:
- No recurrence, purely attention-based
- Enables parallel processing of sequences
- State-of-the-art for NLP and increasingly for other modalities
- Self-attention: Query, Key, Value from same input

### Autoencoders
Unsupervised learning architecture:
- Encoder: Compresses input to latent representation
- Decoder: Reconstructs input from latent
- Loss: Reconstruction error
- Applications: Dimensionality reduction, denoising, anomaly detection

### Generative Adversarial Networks (GANs)
Two networks in competition:
- **Generator**: Creates fake samples from noise
- **Discriminator**: Distinguishes real from fake
- Train alternately until generator produces realistic samples
- Applications: Image generation, style transfer, data augmentation

---

## Key Takeaways

1. **Deep learning** learns hierarchical representations through multiple layers
2. **Activation functions** introduce non-linearity, enabling complex pattern learning
3. **Backpropagation** efficiently computes gradients for weight updates
4. **CNNs** exploit spatial structure in images through convolutions and parameter sharing
5. **RNNs/LSTMs** process sequences while maintaining temporal context
6. **Regularization** (dropout, batch norm, L1/L2) prevents overfitting
7. **Optimization** choices (Adam, SGD) significantly impact training
8. **Advanced architectures** (Transformers, Attention, ResNet) enable state-of-the-art performance
9. **Generalization** is the ultimate goal—test performance matters, not training performance

---

## Further Learning Resources

- **Books**: "Deep Learning" by Goodfellow, Bengio, and Courville
- **Courses**: Fast.ai, Stanford CS231n, Andrew Ng's Deep Learning Specialization
- **Frameworks**: PyTorch and TensorFlow for implementation
- **Papers**: arXiv papers for latest research on architectures and techniques
