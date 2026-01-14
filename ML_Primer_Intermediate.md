# Machine Learning Primer - Intermediate Level

## 1. Fundamentals and Problem Formulation

### What is Machine Learning?
Machine learning is the practice of using algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed for specific tasks. The core idea: improve performance on a task through experience.

### Types of Learning

**Supervised Learning**: Learning from labeled data (input-output pairs)
- Goal: Learn a mapping function f: X → Y
- Examples: Classification, regression

**Unsupervised Learning**: Learning from unlabeled data
- Goal: Discover hidden structure or patterns
- Examples: Clustering, dimensionality reduction

**Reinforcement Learning**: Learning through interaction with an environment
- Goal: Learn a policy to maximize cumulative reward
- Examples: Game playing, robotics

### Problem Formulation

**Training Set**: D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- xᵢ ∈ ℝᵈ: feature vector (d features)
- yᵢ: target/label (continuous for regression, discrete for classification)

**Hypothesis Space (H)**: Set of all possible models we consider

**Learning Algorithm**: Method to select best hypothesis h* ∈ H

**Objective**: Minimize loss function L(h) over training data while generalizing to unseen data

## 2. Supervised Learning

### 2.1 Linear Regression

**Model**: h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₐxₐ = wᵀx

**Loss Function**: Mean Squared Error (MSE)
- L(w) = (1/n) Σᵢ (yᵢ - wᵀxᵢ)²

**Solution**: Closed-form using Normal Equations
- w* = (XᵀX)⁻¹Xᵀy

**Assumptions**:
- Linear relationship between features and target
- Independence of errors
- Homoscedasticity (constant variance)
- Normality of errors

**Extensions**:
- Polynomial regression: Add polynomial features
- Ridge regression (L2 regularization): L(w) + λ||w||²
- Lasso regression (L1 regularization): L(w) + λ||w||₁

### 2.2 Logistic Regression

**Model**: P(y=1|x) = σ(wᵀx) where σ(z) = 1/(1+e⁻ᶻ)

**Loss Function**: Binary Cross-Entropy
- L(w) = -(1/n) Σᵢ [yᵢ log(h(xᵢ)) + (1-yᵢ) log(1-h(xᵢ))]

**Decision Boundary**: Linear in feature space

**Multi-class Extension**: Softmax regression
- P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)

### 2.3 Decision Trees

**Structure**: Tree where nodes represent feature tests, branches represent outcomes, leaves represent predictions

**Splitting Criteria**:
- Classification: Gini impurity or Entropy (Information Gain)
  - Gini: G = 1 - Σₖ pₖ²
  - Entropy: H = -Σₖ pₖ log(pₖ)
- Regression: Variance reduction

**Advantages**: Non-linear, interpretable, handles mixed data types

**Disadvantages**: Prone to overfitting, unstable (high variance)

**Pruning**: Pre-pruning (max depth, min samples) or post-pruning to reduce overfitting

### 2.4 Support Vector Machines (SVM)

**Concept**: Find hyperplane that maximizes margin between classes

**Hard Margin**: For linearly separable data
- Maximize 2/||w|| subject to yᵢ(wᵀxᵢ + b) ≥ 1

**Soft Margin**: For non-separable data (C parameter controls trade-off)
- Allow some misclassifications with slack variables ξᵢ

**Kernel Trick**: Map data to higher dimensions implicitly
- Linear: K(x, x') = xᵀx'
- RBF/Gaussian: K(x, x') = exp(-γ||x-x'||²)
- Polynomial: K(x, x') = (xᵀx' + c)ᵈ

**Advantages**: Effective in high dimensions, memory efficient

**Disadvantages**: Slow for large datasets, sensitive to feature scaling

### 2.5 k-Nearest Neighbors (k-NN)

**Algorithm**: Predict based on k closest training examples

**Distance Metrics**:
- Euclidean: √(Σᵢ(xᵢ-yᵢ)²)
- Manhattan: Σᵢ|xᵢ-yᵢ|
- Minkowski: (Σᵢ|xᵢ-yᵢ|ᵖ)^(1/p)

**Classification**: Majority vote among k neighbors

**Regression**: Average of k neighbors' values

**Choosing k**: Small k → low bias, high variance; Large k → high bias, low variance

**Advantages**: Simple, non-parametric, no training phase

**Disadvantages**: Computationally expensive at prediction, sensitive to irrelevant features

## 3. Unsupervised Learning

### 3.1 K-Means Clustering

**Algorithm**:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Objective**: Minimize within-cluster sum of squares (WCSS)
- J = Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

**Choosing k**: Elbow method (plot WCSS vs k) or Silhouette analysis

**Limitations**: Assumes spherical clusters, sensitive to initialization (use k-means++), requires k specification

### 3.2 Hierarchical Clustering

**Agglomerative (Bottom-up)**:
1. Start with each point as a cluster
2. Merge closest clusters iteratively
3. Continue until single cluster

**Linkage Criteria**:
- Single: min distance between points in clusters
- Complete: max distance
- Average: average distance
- Ward: minimize variance when merging

**Output**: Dendrogram showing hierarchical structure

**Advantages**: No need to specify k, provides hierarchy

**Disadvantages**: Computationally expensive O(n³), not scalable

### 3.3 Principal Component Analysis (PCA)

**Goal**: Find orthogonal directions of maximum variance

**Method**:
1. Standardize data (mean=0, variance=1)
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues
4. Sort by eigenvalue (variance explained)
5. Project data onto top k eigenvectors

**Variance Explained**: Choose k such that cumulative variance ≥ threshold (e.g., 95%)

**Applications**: Dimensionality reduction, visualization, noise reduction, feature extraction

**Limitations**: Linear method, assumes orthogonal components, sensitive to scaling

### 3.4 Other Methods

**t-SNE**: Non-linear dimensionality reduction for visualization
- Preserves local structure
- Computationally expensive, non-deterministic

**DBSCAN**: Density-based clustering
- Can find arbitrary shaped clusters
- Automatically identifies outliers
- Parameters: eps (radius), min_samples

**Gaussian Mixture Models (GMM)**: Probabilistic clustering
- Assumes data comes from mixture of Gaussians
- Uses EM algorithm
- Provides soft assignments (probabilities)

## 4. Model Evaluation and Validation

### 4.1 Train-Test Split

**Purpose**: Assess generalization to unseen data

**Typical Split**: 70-80% train, 20-30% test

**Stratification**: Maintain class proportions in classification

### 4.2 Cross-Validation

**k-Fold CV**:
1. Split data into k folds
2. Train on k-1 folds, validate on remaining fold
3. Repeat k times, average results

**Benefits**: Uses all data for training and validation, reduces variance in performance estimate

**Variants**:
- Stratified k-fold: Maintains class proportions
- Leave-One-Out CV (LOOCV): k=n (expensive but unbiased)
- Time Series Split: Respects temporal ordering

### 4.3 Classification Metrics

**Confusion Matrix**:
- True Positives (TP), True Negatives (TN)
- False Positives (FP), False Negatives (FN)

**Accuracy**: (TP + TN) / Total
- Misleading with imbalanced classes

**Precision**: TP / (TP + FP)
- Of predicted positives, how many are correct?

**Recall/Sensitivity**: TP / (TP + FN)
- Of actual positives, how many detected?

**F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean, balances precision and recall

**ROC Curve**: True Positive Rate vs False Positive Rate
- AUC (Area Under Curve): Single metric, 0.5 = random, 1.0 = perfect

**Precision-Recall Curve**: Better for imbalanced datasets

### 4.4 Regression Metrics

**Mean Squared Error (MSE)**: (1/n) Σᵢ (yᵢ - ŷᵢ)²
- Heavily penalizes large errors

**Root Mean Squared Error (RMSE)**: √MSE
- Same units as target variable

**Mean Absolute Error (MAE)**: (1/n) Σᵢ |yᵢ - ŷᵢ|
- More robust to outliers

**R² Score**: 1 - (SS_res / SS_tot)
- Proportion of variance explained
- Range: (-∞, 1], 1 = perfect fit

**Adjusted R²**: Penalizes additional features
- Useful for model comparison

### 4.5 Bias-Variance Tradeoff

**Total Error** = Bias² + Variance + Irreducible Error

**Bias**: Error from wrong assumptions
- High bias → underfitting
- Simple models tend to have high bias

**Variance**: Error from sensitivity to training data fluctuations
- High variance → overfitting
- Complex models tend to have high variance

**Tradeoff**: Increasing model complexity reduces bias but increases variance

**Goal**: Find sweet spot that minimizes total error

## 5. Optimization and Training

### 5.1 Gradient Descent

**Concept**: Iteratively move in direction of steepest descent

**Update Rule**: w_{t+1} = w_t - η ∇L(w_t)
- η: learning rate (step size)
- ∇L: gradient of loss function

**Batch Gradient Descent**: Use entire dataset per update
- Stable but slow for large datasets

**Stochastic Gradient Descent (SGD)**: Use single sample per update
- Fast, noisy, can escape local minima

**Mini-batch Gradient Descent**: Use small batch per update
- Balance between batch and SGD
- Typical batch sizes: 32, 64, 128, 256

### 5.2 Learning Rate

**Too Large**: Oscillation, divergence

**Too Small**: Slow convergence, stuck in local minima

**Adaptive Methods**:
- Momentum: Accumulate velocity, smooth updates
- AdaGrad: Adapt learning rate per parameter
- RMSProp: Use moving average of squared gradients
- Adam: Combines momentum and RMSProp (popular choice)

**Learning Rate Schedules**:
- Step decay: Reduce by factor every n epochs
- Exponential decay: η_t = η_0 × e^(-kt)
- Cosine annealing: Smooth reduction

### 5.3 Regularization

**Purpose**: Prevent overfitting by constraining model complexity

**L2 Regularization (Ridge)**:
- Loss: L(w) + λ||w||²
- Shrinks weights toward zero
- Prefers smaller, distributed weights

**L1 Regularization (Lasso)**:
- Loss: L(w) + λ||w||₁
- Produces sparse solutions (some weights = 0)
- Performs feature selection

**Elastic Net**: Combines L1 and L2
- Loss: L(w) + λ₁||w||₁ + λ₂||w||²

**Dropout**: Randomly deactivate neurons during training (neural networks)

**Early Stopping**: Stop training when validation error increases

**Data Augmentation**: Artificially expand training set

## 6. Advanced Topics

### 6.1 Ensemble Methods

**Concept**: Combine multiple models for better performance

**Bagging (Bootstrap Aggregating)**:
- Train models on random subsets with replacement
- Average predictions (regression) or vote (classification)
- Reduces variance
- Example: Random Forest

**Random Forest**:
- Ensemble of decision trees
- Each tree trained on bootstrap sample
- Each split considers random subset of features
- Robust, handles high-dimensional data, provides feature importance

**Boosting**:
- Train models sequentially
- Each model focuses on errors of previous models
- Reduces bias and variance
- Examples: AdaBoost, Gradient Boosting, XGBoost, LightGBM

**Gradient Boosting**:
- Build trees to predict residuals of previous model
- Final prediction: sum of all trees
- Powerful but prone to overfitting
- Hyperparameters: n_estimators, learning_rate, max_depth

**Stacking**:
- Train meta-model on predictions of base models
- Can combine diverse model types

### 6.2 Neural Networks Basics

**Structure**: Layers of interconnected neurons
- Input layer: Receives features
- Hidden layers: Learn representations
- Output layer: Produces predictions

**Neuron**: z = wᵀx + b, a = σ(z)
- z: weighted sum (pre-activation)
- σ: activation function
- a: activation (output)

**Activation Functions**:
- Sigmoid: σ(z) = 1/(1+e⁻ᶻ) [0,1]
- Tanh: tanh(z) [-1,1]
- ReLU: max(0, z) [most common]
- Leaky ReLU: max(0.01z, z)

**Forward Propagation**: Compute outputs layer by layer

**Backpropagation**: Compute gradients using chain rule
- Efficient gradient computation
- Update weights via gradient descent

**Universal Approximation Theorem**: A network with sufficient hidden units can approximate any continuous function

### 6.3 Feature Engineering

**Importance**: Often more impactful than model selection

**Techniques**:
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Encoding categorical variables: One-hot, label encoding, target encoding
- Feature creation: Polynomial features, interactions, domain-specific features
- Feature selection: Filter (correlation), wrapper (RFE), embedded (Lasso)
- Handling missing values: Imputation, indicator variables
- Binning: Convert continuous to categorical

**Feature Importance**: Use tree-based models or permutation importance

## 7. Practical Considerations and Best Practices

### 7.1 Data Preprocessing

**Must Do**:
- Handle missing values before splitting
- Scale features (especially for distance-based and gradient-based methods)
- Encode categorical variables
- Check for data leakage

**Data Leakage**: Information from test set influencing training
- Fit preprocessing on training data only
- Apply same transformation to test data

### 7.2 Hyperparameter Tuning

**Grid Search**: Exhaustively try all combinations
- Comprehensive but expensive

**Random Search**: Sample random combinations
- More efficient, often performs similarly

**Bayesian Optimization**: Intelligently explore hyperparameter space
- Uses previous results to guide search

**Important Hyperparameters by Model**:
- Decision Trees: max_depth, min_samples_split
- Random Forest: n_estimators, max_features
- SVM: C, gamma (for RBF kernel)
- k-NN: k, distance metric
- Neural Networks: learning_rate, batch_size, architecture

### 7.3 Handling Imbalanced Data

**Problem**: Majority class dominates, model ignores minority

**Solutions**:
- Resampling: Oversample minority (SMOTE) or undersample majority
- Class weights: Penalize misclassification of minority more
- Anomaly detection: Treat as one-class problem
- Ensemble methods: Use balanced bootstrap samples
- Evaluation: Focus on F1, precision-recall, not accuracy

### 7.4 Model Selection Workflow

1. **Understand the problem**: Classification/regression, data size, feature types
2. **Establish baseline**: Simple model (e.g., logistic regression, decision tree)
3. **Try multiple models**: Compare performance
4. **Tune hyperparameters**: For best performing models
5. **Ensemble**: Combine top models if needed
6. **Validate**: Use holdout test set for final evaluation
7. **Interpret**: Understand what model learned (feature importance, SHAP values)

### 7.5 Common Pitfalls

- Using test data for any decision (hyperparameter tuning, feature selection)
- Not scaling features for distance/gradient-based methods
- Ignoring class imbalance
- Overfitting to validation set through excessive tuning
- Not checking for data leakage
- Using accuracy for imbalanced classification
- Extrapolating beyond training data range
- Ignoring domain knowledge

### 7.6 Model Interpretability

**Why**: Trust, debugging, compliance, scientific insight

**Methods**:
- Feature importance: Tree-based models, permutation importance
- Partial Dependence Plots: Show effect of feature on prediction
- SHAP values: Unified approach, game-theoretic
- LIME: Local interpretable model-agnostic explanations
- Coefficient inspection: Linear models

## Summary

Machine learning is an iterative process of:
1. Formulating the problem
2. Preparing and understanding data
3. Selecting and training models
4. Evaluating and tuning
5. Deploying and monitoring

Success requires understanding both the mathematics and practical considerations. Start simple, iterate, and always validate properly.
