# Loss Functions and Activation Functions

## Loss Functions

Loss functions measure how well a model's predictions match the actual targets. The choice depends on your task:

### Regression Tasks
- **Mean Squared Error (MSE)**: Standard choice for continuous predictions. Penalizes larger errors more heavily (quadratic penalty). Use when you want outliers to significantly impact training.
- **Mean Absolute Error (MAE)**: Less sensitive to outliers than MSE. Better when you have noisy data or don't want extreme errors to dominate.
- **Huber Loss**: Hybrid approach combining MSE and MAE—acts like MSE for small errors and MAE for large ones. Good for robust regression with some outliers.

### Classification Tasks
- **Binary Crossentropy**: For binary classification (two classes). Measures divergence between predicted and true probability distributions.
- **Categorical Crossentropy**: For multi-class classification (3+ classes). Each output represents probability of a class.
- **Sparse Categorical Crossentropy**: Same as categorical crossentropy but accepts integer class labels instead of one-hot encoded vectors. More memory efficient.
- **Focal Loss**: For imbalanced datasets. Down-weights easy examples and focuses on hard negatives. Useful when one class is much more frequent.

### Other Scenarios
- **KL Divergence**: When you want to match probability distributions directly (e.g., knowledge distillation).
- **Contrastive Loss**: For similarity learning tasks (e.g., learning embeddings where similar samples are close).
- **Triplet Loss**: For metric learning where you want to push similar samples together and dissimilar ones apart.

## Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns:

### Hidden Layer Activations
- **ReLU (Rectified Linear Unit)**: Default choice for most deep networks. Fast to compute, helps prevent vanishing gradient problem. Returns max(0, x).
- **Leaky ReLU**: Addresses ReLU's "dying ReLU" problem where neurons can become inactive. Allows small negative values through (e.g., 0.01x when x < 0).
- **ELU (Exponential Linear Unit)**: Smoother than ReLU, can output negative values, helps with mean activation being closer to zero. Slightly more computationally expensive.
- **GELU (Gaussian Error Linear Unit)**: Used in modern transformer models. Smooth, probabilistic interpretation. Good for state-of-the-art language models.
- **Tanh**: Outputs values in [-1, 1]. Good for hidden layers but slower than ReLU. Useful when you need centered outputs.
- **Sigmoid**: Outputs [0, 1]. Generally avoided in hidden layers (vanishing gradients), but useful elsewhere.

### Output Layer Activations
- **Sigmoid**: For binary classification—outputs probability in [0, 1].
- **Softmax**: For multi-class classification—converts raw scores to probability distribution over all classes.
- **Linear (No activation)**: For regression—no constraint on output values.
- **ReLU/Softplus**: For regression with positive-only outputs (e.g., predicting counts, intensities).

### Special Use Cases
- **Swish/SiLU**: Self-gated activation used in EfficientNets and other modern architectures. Provides better performance than ReLU in some cases.
- **Mish**: Smooth activation with good empirical results. Slightly slower but sometimes better accuracy.

## Practical Selection Guide

| Task | Loss Function | Hidden Activation | Output Activation |
|------|--------------|-------------------|-------------------|
| Binary Classification | Binary Crossentropy | ReLU | Sigmoid |
| Multi-class Classification | Categorical Crossentropy | ReLU | Softmax |
| Regression | MSE or MAE | ReLU | Linear |
| Imbalanced Classification | Focal Loss | ReLU | Sigmoid/Softmax |
| Embedding/Similarity Learning | Triplet/Contrastive | ReLU or GELU | Linear or Normalized |

**Key Takeaway**: Match your loss function to your task (what you're predicting) and choose activations that help your network train effectively without gradient issues.
