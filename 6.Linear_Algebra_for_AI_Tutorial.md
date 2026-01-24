# Linear Algebra for Artificial Intelligence: A Comprehensive Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Scalars, Vectors, and Matrices](#scalars-vectors-and-matrices)
3. [Vector Operations](#vector-operations)
4. [Matrix Operations](#matrix-operations)
5. [Linear Transformations](#linear-transformations)
6. [Matrix Decompositions](#matrix-decompositions)
7. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
8. [Norms and Distance Metrics](#norms-and-distance-metrics)
9. [Orthogonality and Projections](#orthogonality-and-projections)
10. [Solving Linear Systems](#solving-linear-systems)
11. [Least Squares and Regression](#least-squares-and-regression)
12. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
13. [Tensor Operations](#tensor-operations)
14. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Introduction

Linear algebra is the mathematical foundation of artificial intelligence and machine learning. It provides the language and tools to:
- Represent and manipulate data
- Understand neural network operations
- Optimize models through gradient descent
- Perform dimensionality reduction
- Analyze data patterns and relationships

---

## Scalars, Vectors, and Matrices

### Scalars
A scalar is a single numerical value.

**Example:**
```
a = 5
b = 3.14
c = -2.7
```

### Vectors
A vector is an ordered array of numbers, representing a point in space or a direction.

**Example - Column Vector:**
```
v = [2]
    [3]
    [5]
```

**Example - Row Vector:**
```
v^T = [2  3  5]
```

**AI Application:** In machine learning, feature vectors represent data points. For example, an image of 28×28 pixels can be flattened into a vector of length 784.

**Python Example:**
```python
import numpy as np

# Creating a vector
v = np.array([2, 3, 5])
print(f"Vector: {v}")
print(f"Shape: {v.shape}")  # (3,)
```

### Matrices
A matrix is a 2D array of numbers arranged in rows and columns.

**Example:**
```
A = [1  2  3]
    [4  5  6]
    [7  8  9]
```

**Notation:** A matrix of m rows and n columns is denoted as A ∈ ℝ^(m×n)

**AI Application:** Weight matrices in neural networks, where each connection between layers is represented by a matrix.

**Python Example:**
```python
import numpy as np

# Creating a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"Matrix:\n{A}")
print(f"Shape: {A.shape}")  # (3, 3)
```

---

## Vector Operations

### Vector Addition
Vectors of the same dimension can be added element-wise.

**Example:**
```
v1 = [1]    v2 = [4]    v1 + v2 = [5]
     [2]         [5]              [7]
     [3]         [6]              [9]
```

**Python Example:**
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = v1 + v2
print(f"v1 + v2 = {result}")  # [5 7 9]
```

### Scalar Multiplication
Multiply each element of a vector by a scalar.

**Example:**
```
c = 3
v = [1]    c·v = [3]
    [2]          [6]
    [4]          [12]
```

**Python Example:**
```python
c = 3
v = np.array([1, 2, 4])
result = c * v
print(f"3 * v = {result}")  # [3 6 12]
```

### Dot Product (Inner Product)
The dot product of two vectors produces a scalar.

**Formula:** v·w = v₁w₁ + v₂w₂ + ... + vₙwₙ

**Example:**
```
v = [1]    w = [4]
    [2]        [5]
    [3]        [6]

v·w = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**AI Application:** Computing similarity between vectors (e.g., word embeddings), calculating neuron activations.

**Python Example:**
```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
dot_product = np.dot(v, w)
print(f"v·w = {dot_product}")  # 32
```

### Cross Product
The cross product of two 3D vectors produces another vector perpendicular to both.

**Example:**
```
v = [1]    w = [4]
    [2]        [5]
    [3]        [6]

v × w = [2×6 - 3×5]   = [-3]
        [3×4 - 1×6]     [6]
        [1×5 - 2×4]     [-3]
```

**Python Example:**
```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
cross_product = np.cross(v, w)
print(f"v × w = {cross_product}")  # [-3  6 -3]
```

---

## Matrix Operations

### Matrix Addition
Add corresponding elements of matrices with the same dimensions.

**Example:**
```
A = [1  2]    B = [5  6]    A + B = [6   8]
    [3  4]        [7  8]            [10  12]
```

**Python Example:**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A + B
print(f"A + B =\n{result}")
```

### Scalar-Matrix Multiplication
Multiply every element by a scalar.

**Example:**
```
c = 2
A = [1  2]    c·A = [2  4]
    [3  4]          [6  8]
```

### Matrix Multiplication
Multiply matrices by taking dot products of rows and columns.

**Rule:** For A(m×n) × B(n×p), result is C(m×p)

**Example:**
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

C = A×B = [1×5+2×7  1×6+2×8]  = [19  22]
          [3×5+4×7  3×6+4×8]    [43  50]
```

**AI Application:** Core operation in neural networks - computing layer outputs from inputs.

**Python Example:**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.matmul(A, B)  # or A @ B
print(f"A × B =\n{C}")
```

### Matrix Transpose
Flip a matrix over its diagonal (rows become columns).

**Example:**
```
A = [1  2  3]    A^T = [1  4]
    [4  5  6]          [2  5]
                       [3  6]
```

**Python Example:**
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A_T = A.T
print(f"A^T =\n{A_T}")
```

### Matrix Inverse
The inverse A⁻¹ satisfies: A × A⁻¹ = I (identity matrix)

**Example:**
```
A = [4  7]    A⁻¹ = [0.6  -0.7]
    [2  6]          [-0.2  0.4]

Verify: A × A⁻¹ = [1  0]
                  [0  1]
```

**AI Application:** Solving systems of equations, computing optimal parameters in linear regression.

**Python Example:**
```python
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)
print(f"A⁻¹ =\n{A_inv}")

# Verify
I = np.matmul(A, A_inv)
print(f"A × A⁻¹ =\n{I}")  # Identity matrix (approximately)
```

### Determinant
A scalar value that provides information about the matrix properties.

**Example (2×2):**
```
A = [a  b]    det(A) = ad - bc
    [c  d]
```

**Example:**
```
A = [4  7]    det(A) = 4×6 - 7×2 = 24 - 14 = 10
    [2  6]
```

**AI Application:** Checking matrix invertibility, volume transformations in space.

**Python Example:**
```python
A = np.array([[4, 7], [2, 6]])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")  # 10.0
```

---

## Linear Transformations

A linear transformation T: ℝⁿ → ℝᵐ maps vectors from one space to another while preserving:
1. Addition: T(v + w) = T(v) + T(w)
2. Scalar multiplication: T(cv) = cT(v)

Every linear transformation can be represented as matrix multiplication.

**Example - Rotation Matrix (2D):**
Rotate a vector by angle θ counterclockwise:

```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

**Example - Rotate by 90°:**
```
R(90°) = [0  -1]
         [1   0]

v = [1]    R(90°)v = [0  -1][1] = [-2]
    [2]              [1   0][2]   [1]
```

**Python Example:**
```python
import numpy as np

# Rotation by 90 degrees
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([1, 2])
v_rotated = R @ v
print(f"Original: {v}")
print(f"Rotated: {v_rotated}")
```

**Example - Scaling Transformation:**
```
S = [2  0]    # Scale x by 2, y by 3
    [0  3]

v = [1]    Sv = [2  0][1] = [2]
    [2]        [0  3][2]   [6]
```

**AI Application:** Convolutional layers apply learned transformations, data augmentation uses geometric transformations.

---

## Matrix Decompositions

### LU Decomposition
Decompose A into lower triangular (L) and upper triangular (U) matrices: A = LU

**Example:**
```
A = [2  1]  = [1    0][2  1]
    [4  3]    [2    1][0  1]
              L        U
```

**AI Application:** Efficiently solving systems of linear equations.

**Python Example:**
```python
from scipy.linalg import lu

A = np.array([[2, 1], [4, 3]])
P, L, U = lu(A)
print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"Verification A = LU:\n{L @ U}")
```

### QR Decomposition
Decompose A into orthogonal matrix (Q) and upper triangular matrix (R): A = QR

**Properties:**
- Q^T Q = I (Q is orthogonal)
- R is upper triangular

**AI Application:** Stable computation of least squares, numerical stability in algorithms.

**Python Example:**
```python
A = np.array([[1, 2], [3, 4], [5, 6]])
Q, R = np.linalg.qr(A)
print(f"Q =\n{Q}")
print(f"R =\n{R}")
print(f"Q^T Q =\n{Q.T @ Q}")  # Should be close to identity
```

### Cholesky Decomposition
For positive definite symmetric matrix A: A = LL^T

**Example:**
```
A = [4   12]  = [2  0][2  6]
    [12  37]    [6  1][0  1]
                L     L^T
```

**AI Application:** Covariance matrices in Gaussian processes, optimization algorithms.

**Python Example:**
```python
A = np.array([[4, 12], [12, 37]])
L = np.linalg.cholesky(A)
print(f"L =\n{L}")
print(f"LL^T =\n{L @ L.T}")
```

---

## Eigenvalues and Eigenvectors

For a square matrix A, an eigenvector v and eigenvalue λ satisfy: **Av = λv**

The eigenvector's direction doesn't change under the transformation, only its magnitude (by λ).

**Example:**
```
A = [3  1]
    [0  2]

Eigenvalues: λ₁ = 3, λ₂ = 2

Eigenvector for λ₁=3: v₁ = [1]
                           [0]

Eigenvector for λ₂=2: v₂ = [1]
                           [-1]

Verify: A v₁ = [3  1][1]  = [3]  = 3[1]  = λ₁v₁ ✓
              [0  2][0]    [0]     [0]
```

**AI Applications:**
- PCA (dimensionality reduction)
- Analyzing neural network dynamics
- PageRank algorithm
- Spectral clustering

**Python Example:**
```python
A = np.array([[3, 1], [0, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify for first eigenvalue
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"\nAv₁ = {A @ v1}")
print(f"λ₁v₁ = {lambda1 * v1}")
```

### Eigendecomposition
A diagonalizable matrix can be written as: **A = QΛQ⁻¹**

Where:
- Q: matrix of eigenvectors (columns)
- Λ: diagonal matrix of eigenvalues

**Example:**
```
A = [3  1]  = [1   1][3  0][1  1]⁻¹
    [0  2]    [0  -1][0  2][0 -1]
              Q      Λ     Q⁻¹
```

---

## Singular Value Decomposition (SVD)

The most important matrix decomposition for AI. Any matrix A (m×n) can be decomposed:

**A = UΣV^T**

Where:
- U (m×m): left singular vectors (orthogonal)
- Σ (m×n): diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V (n×n): right singular vectors (orthogonal)

**Example:**
```
A = [1  2]
    [3  4]
    [5  6]

A = UΣV^T can be computed numerically
```

**AI Applications:**
- Dimensionality reduction
- Data compression
- Recommender systems (matrix factorization)
- Image compression
- Latent Semantic Analysis (LSA)
- Pseudoinverse computation

**Python Example:**
```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, VT = np.linalg.svd(A)

print(f"U shape: {U.shape}")
print(f"Singular values: {S}")
print(f"V^T shape: {VT.shape}")

# Reconstruct A
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(S)
A_reconstructed = U @ Sigma @ VT
print(f"\nOriginal A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
```

**Low-Rank Approximation:**
Keep only top k singular values for compression:

```python
# Keep only top 1 singular value
k = 1
Sigma_k = np.zeros((A.shape[0], A.shape[1]))
Sigma_k[0, 0] = S[0]
A_approx = U @ Sigma_k @ VT
print(f"\nRank-{k} approximation:\n{A_approx}")
```

---

## Norms and Distance Metrics

Norms measure the "size" or "length" of vectors and matrices.

### Vector Norms

**L1 Norm (Manhattan Distance):**
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

**Example:**
```
v = [-3, 4]
||v||₁ = |-3| + |4| = 3 + 4 = 7
```

**L2 Norm (Euclidean Distance):**
```
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

**Example:**
```
v = [3, 4]
||v||₂ = √(3² + 4²) = √(9 + 16) = √25 = 5
```

**L∞ Norm (Maximum Norm):**
```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

**Example:**
```
v = [-3, 4, 2]
||v||∞ = max(3, 4, 2) = 4
```

**AI Applications:**
- Regularization (L1 for sparsity, L2 for weight decay)
- Loss functions
- Distance metrics for clustering
- Gradient clipping

**Python Example:**
```python
v = np.array([3, 4])

l1_norm = np.linalg.norm(v, ord=1)
l2_norm = np.linalg.norm(v, ord=2)
linf_norm = np.linalg.norm(v, ord=np.inf)

print(f"L1 norm: {l1_norm}")    # 7.0
print(f"L2 norm: {l2_norm}")    # 5.0
print(f"L∞ norm: {linf_norm}")  # 4.0
```

### Distance Metrics

**Euclidean Distance:**
```
d(v, w) = ||v - w||₂
```

**Cosine Similarity:**
Measures angle between vectors (useful for high-dimensional data):
```
cos(θ) = (v·w) / (||v||₂ × ||w||₂)
```

**Example:**
```
v = [1, 2, 3]
w = [2, 4, 6]

cos(θ) = (1×2 + 2×4 + 3×6) / (√14 × √56)
       = 28 / 28 = 1  (vectors are parallel)
```

**Python Example:**
```python
from sklearn.metrics.pairwise import cosine_similarity

v = np.array([[1, 2, 3]])
w = np.array([[2, 4, 6]])

similarity = cosine_similarity(v, w)
print(f"Cosine similarity: {similarity[0][0]}")  # 1.0
```

---

## Orthogonality and Projections

### Orthogonal Vectors
Two vectors are orthogonal if their dot product is zero: **v·w = 0**

**Example:**
```
v = [1]     w = [-2]
    [2]         [1]

v·w = 1×(-2) + 2×1 = -2 + 2 = 0  ✓ Orthogonal
```

### Orthonormal Vectors
Vectors that are orthogonal and have unit length (||v||₂ = 1).

**Example:**
```
v = [1/√2]    w = [-1/√2]
    [1/√2]        [1/√2]

v·w = 0  ✓ Orthogonal
||v||₂ = 1, ||w||₂ = 1  ✓ Unit length
```

### Orthogonal Matrices
A square matrix Q is orthogonal if: **Q^T Q = I** (or Q^T = Q⁻¹)

**Properties:**
- Preserves lengths: ||Qv||₂ = ||v||₂
- Preserves angles
- Columns are orthonormal vectors

### Vector Projection
Project vector v onto vector u:

**Formula:**
```
proj_u(v) = (v·u / ||u||²) × u
```

**Example:**
```
v = [3]     u = [1]
    [4]         [0]

proj_u(v) = (3×1 + 4×0) / (1² + 0²) × [1] = [3]
                                       [0]   [0]
```

**AI Application:** Feature extraction, removing components (e.g., in Gram-Schmidt orthogonalization).

**Python Example:**
```python
v = np.array([3, 4])
u = np.array([1, 0])

projection = (np.dot(v, u) / np.dot(u, u)) * u
print(f"Projection of v onto u: {projection}")  # [3. 0.]
```

### Gram-Schmidt Orthogonalization
Convert a set of linearly independent vectors into orthonormal vectors.

**Python Example:**
```python
def gram_schmidt(vectors):
    """Convert vectors to orthonormal basis"""
    orthonormal = []
    for v in vectors:
        # Subtract projections onto previous vectors
        w = v.copy()
        for u in orthonormal:
            w = w - np.dot(v, u) * u
        # Normalize
        w = w / np.linalg.norm(w)
        orthonormal.append(w)
    return np.array(orthonormal)

# Example
vectors = [np.array([1., 1., 0.]), 
           np.array([1., 0., 1.])]
orthonormal = gram_schmidt(vectors)
print(f"Orthonormal vectors:\n{orthonormal}")

# Verify orthogonality
print(f"Dot product: {np.dot(orthonormal[0], orthonormal[1])}")  # ~0
```

---

## Solving Linear Systems

A system of linear equations can be written as: **Ax = b**

**Example:**
```
2x + y = 5
x - y = 1

Matrix form: [2   1][x] = [5]
             [1  -1][y]   [1]
```

### Direct Solution
If A is invertible: **x = A⁻¹b**

**Example:**
```
A = [2   1]    b = [5]
    [1  -1]        [1]

x = A⁻¹b = [1/3  1/3][5] = [2]
           [1/3 -2/3][1]   [1]

Solution: x = 2, y = 1
```

**Python Example:**
```python
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])

x = np.linalg.solve(A, b)  # More stable than computing A⁻¹
print(f"Solution: {x}")  # [2. 1.]

# Verify
print(f"Ax = {A @ x}")  # Should equal b
```

### Iterative Methods

**Gradient Descent for Ax = b:**
Minimize f(x) = ||Ax - b||²

**Python Example:**
```python
def gradient_descent_linear(A, b, lr=0.01, iterations=1000):
    x = np.zeros(A.shape[1])
    for _ in range(iterations):
        gradient = 2 * A.T @ (A @ x - b)
        x = x - lr * gradient
    return x

A = np.array([[2., 1.], [1., -1.]])
b = np.array([5., 1.])

x = gradient_descent_linear(A, b)
print(f"Solution via gradient descent: {x}")
```

---

## Least Squares and Regression

When Ax = b has no exact solution (overdetermined system), find the best approximate solution.

**Normal Equations:** Minimize ||Ax - b||²

**Solution:** **x = (A^T A)⁻¹ A^T b**

The term (A^T A)⁻¹ A^T is called the **pseudoinverse** of A.

**Example - Linear Regression:**
```
Data points: (1, 2), (2, 3), (3, 5)
Fit line: y = mx + c

[1  1][m] ≈ [2]
[2  1][c]   [3]
[3  1]      [5]

A         x     b
```

**Python Example:**
```python
# Data points
X = np.array([[1, 1], [2, 1], [3, 1]])  # [x, bias]
y = np.array([2, 3, 5])

# Solve using least squares
params = np.linalg.lstsq(X, y, rcond=None)[0]
m, c = params
print(f"Best fit line: y = {m:.2f}x + {c:.2f}")

# Using normal equations
params_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Via normal equations: {params_normal}")

# Using pseudoinverse
params_pinv = np.linalg.pinv(X) @ y
print(f"Via pseudoinverse: {params_pinv}")
```

**AI Application:** Foundation of linear regression, training linear models.

---

## Principal Component Analysis (PCA)

PCA finds directions of maximum variance in data using eigenvalue decomposition.

**Algorithm:**
1. Center the data: X' = X - mean(X)
2. Compute covariance matrix: C = (X'^T X') / (n-1)
3. Find eigenvectors and eigenvalues of C
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors as principal components
6. Project data: X_reduced = X' × V_k

**Example:**
```
Data points: [1, 2], [2, 3], [3, 5], [4, 6]

Step 1: Center data
Mean = [2.5, 4]
Centered: [[-1.5, -2], [-0.5, -1], [0.5, 1], [1.5, 2]]

Step 2-4: Compute PCA (shown in code below)
```

**Python Example:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 samples, 5 features

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Manual PCA implementation
def manual_pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k eigenvectors
    principal_components = eigenvectors[:, :n_components]
    
    # Project data
    X_reduced = X_centered @ principal_components
    
    return X_reduced, eigenvalues, eigenvectors

X_reduced_manual, eigenvalues, eigenvectors = manual_pca(X, 2)
print(f"\nManual PCA result shape: {X_reduced_manual.shape}")
```

**AI Applications:**
- Dimensionality reduction for visualization
- Feature extraction
- Data compression
- Removing noise from data
- Preprocessing for machine learning

---

## Tensor Operations

Tensors are multi-dimensional arrays generalizing scalars (0D), vectors (1D), and matrices (2D).

**Notation:**
- 0D tensor: scalar
- 1D tensor: vector
- 2D tensor: matrix
- 3D tensor: cube of numbers
- nD tensor: n-dimensional array

**Example - 3D Tensor:**
```
Shape: (2, 3, 4) means 2 matrices of size 3×4

T[0] = [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]

T[1] = [[13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]]
```

**Python Example:**
```python
# Creating tensors
tensor_3d = np.random.randn(2, 3, 4)
print(f"3D Tensor shape: {tensor_3d.shape}")

tensor_4d = np.random.randn(10, 32, 32, 3)  # 10 RGB images of 32×32
print(f"4D Tensor shape: {tensor_4d.shape}")
```

### Tensor Operations

**Reshaping:**
```python
x = np.arange(12)  # [0, 1, 2, ..., 11]
x_reshaped = x.reshape(3, 4)
print(f"Reshaped to 3×4:\n{x_reshaped}")

x_3d = x.reshape(2, 2, 3)
print(f"Reshaped to 2×2×3:\n{x_3d}")
```

**Flattening:**
```python
x = np.random.randn(28, 28)  # Image
x_flat = x.reshape(-1)  # or x.flatten()
print(f"Flattened shape: {x_flat.shape}")  # (784,)
```

**Broadcasting:**
Automatic expansion of dimensions for operations.

```python
# Add vector to each row of matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

result = A + b  # b is broadcast to each row
print(f"Result:\n{result}")
# [[11, 22, 33],
#  [14, 25, 36]]
```

**Tensor Contraction (Einstein Summation):**

```python
# Matrix multiplication using einsum
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.einsum('ij,jk->ik', A, B)  # Equivalent to A @ B

# Batch matrix multiplication
batch_A = np.random.randn(10, 3, 4)
batch_B = np.random.randn(10, 4, 5)
batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"Batch result shape: {batch_C.shape}")  # (10, 3, 5)

# Trace of matrix
A = np.random.randn(5, 5)
trace = np.einsum('ii->', A)  # Sum of diagonal
print(f"Trace: {trace}")
```

**AI Application:** Tensors are the fundamental data structure in deep learning frameworks (TensorFlow, PyTorch).

---

## Applications in Deep Learning

### 1. Neural Network Forward Pass

A simple feedforward layer: **z = Wx + b**, **a = σ(z)**

**Example:**
```
Input: x = [1, 2, 3]
Weights: W = [[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]]
Bias: b = [0.5, 0.5]

z = Wx + b = [[0.1, 0.2, 0.3],    [[1],    [[0.5],
              [0.4, 0.5, 0.6]]  ×   [2],  +  [0.5]]
                                    [3]]

z = [1.4 + 0.5,   = [1.9,
     3.2 + 0.5]      3.7]

a = σ(z)  # Apply activation function
```

**Python Example:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Input
x = np.array([1, 2, 3])
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
b = np.array([0.5, 0.5])

# Forward pass
z = W @ x + b
a = sigmoid(z)

print(f"Linear output z: {z}")
print(f"Activation a: {a}")
```

### 2. Backpropagation (Chain Rule with Matrices)

Compute gradients using the chain rule.

**Example - Single Layer:**
```
Loss: L = (1/2)||y - a||²
where a = σ(Wx + b)

Gradients:
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
      = (a - y) ⊙ σ'(z) × x^T
```

**Python Example:**
```python
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward pass
x = np.array([1, 2, 3])
y = np.array([1, 0])  # True labels
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
b = np.array([0.5, 0.5])

z = W @ x + b
a = sigmoid(z)

# Backward pass
dL_da = a - y  # Gradient of loss w.r.t. activation
da_dz = sigmoid_derivative(z)  # Gradient of activation
dL_dz = dL_da * da_dz  # Element-wise multiply

dL_dW = np.outer(dL_dz, x)  # Gradient w.r.t. weights
dL_db = dL_dz  # Gradient w.r.t. bias

print(f"Gradient dL/dW:\n{dL_dW}")
print(f"Gradient dL/db: {dL_db}")

# Update weights (gradient descent)
learning_rate = 0.1
W = W - learning_rate * dL_dW
b = b - learning_rate * dL_db
```

### 3. Batch Processing

Process multiple samples simultaneously using matrices.

**Example:**
```
Batch of 4 samples, 3 features each:
X = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9],
     [10, 11, 12]]

Shape: (4, 3) = (batch_size, features)

Z = XW^T + b  # W shape: (output_dim, input_dim)
```

**Python Example:**
```python
# Batch processing
batch_size = 4
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
b = np.array([0.5, 0.5])

# Forward pass for entire batch
Z = X @ W.T + b  # Broadcasting b to each sample
A = sigmoid(Z)

print(f"Batch output shape: {A.shape}")  # (4, 2)
print(f"Activations:\n{A}")
```

### 4. Convolutional Operations

Convolutions are matrix multiplications with structured matrices.

**Example - 1D Convolution:**
```
Input: x = [1, 2, 3, 4, 5]
Kernel: k = [1, 0, -1]

Output:
y[0] = 1×1 + 2×0 + 3×(-1) = -2
y[1] = 2×1 + 3×0 + 4×(-1) = -2
y[2] = 3×1 + 4×0 + 5×(-1) = -2
```

**Python Example:**
```python
from scipy.signal import correlate

# 1D Convolution
x = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, 0, -1])

output = correlate(x, kernel, mode='valid')
print(f"Convolution output: {output}")

# 2D Convolution (image)
image = np.random.randn(28, 28)
kernel_2d = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])  # Edge detection

from scipy.signal import convolve2d
edges = convolve2d(image, kernel_2d, mode='valid')
print(f"Edge detection output shape: {edges.shape}")
```

### 5. Attention Mechanism

The attention mechanism in Transformers uses matrix operations.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: dimension of keys
```

**Python Example:**
```python
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention to values
    output = attention_weights @ V
    
    return output, attention_weights

# Example
seq_length = 4
d_model = 8

Q = np.random.randn(seq_length, d_model)
K = np.random.randn(seq_length, d_model)
V = np.random.randn(seq_length, d_model)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Attention output shape: {output.shape}")
print(f"Attention weights:\n{weights}")
print(f"Weights sum to 1: {weights.sum(axis=1)}")  # Each row sums to 1
```

### 6. Gradient Descent Optimization

**Batch Gradient Descent:**
```
θ = θ - α∇J(θ)
where ∇J(θ) = (1/m) X^T(Xθ - y)
```

**Python Example:**
```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Predictions
        predictions = X @ theta
        
        # Compute gradient
        gradient = (1/m) * X.T @ (predictions - y)
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Compute loss (optional)
        if epoch % 100 == 0:
            loss = (1/(2*m)) * np.sum((predictions - y)**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return theta

# Example
X = np.random.randn(100, 3)
true_theta = np.array([2, -1, 0.5])
y = X @ true_theta + np.random.randn(100) * 0.1

learned_theta = gradient_descent(X, y)
print(f"\nTrue parameters: {true_theta}")
print(f"Learned parameters: {learned_theta}")
```

### 7. Word Embeddings and Similarity

**Example - Computing Similarities:**
```python
# Word vectors (simplified)
words = {
    'king': np.array([0.1, 0.3, 0.5]),
    'queen': np.array([0.2, 0.4, 0.5]),
    'man': np.array([0.0, 0.1, 0.3]),
    'woman': np.array([0.1, 0.2, 0.3])
}

# Analogy: king - man + woman ≈ queen
result = words['king'] - words['man'] + words['woman']

# Find most similar word
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

similarities = {}
for word, vec in words.items():
    similarities[word] = cosine_similarity(result, vec)

print("Similarities:")
for word, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {sim:.4f}")
```

---

## Summary: Key Linear Algebra Concepts for AI

| Concept | AI Application |
|---------|---------------|
| **Vectors & Matrices** | Data representation, neural network parameters |
| **Matrix Multiplication** | Neural network forward/backward pass |
| **Transpose** | Backpropagation, gradient computation |
| **Dot Product** | Similarity measures, attention mechanisms |
| **Norms** | Regularization, loss functions |
| **Eigenvalues/Eigenvectors** | PCA, analyzing network dynamics |
| **SVD** | Dimensionality reduction, matrix factorization |
| **Inverse & Pseudoinverse** | Solving linear systems, linear regression |
| **Orthogonality** | Feature decorrelation, stable algorithms |
| **Projections** | Dimensionality reduction, feature extraction |
| **Tensors** | Deep learning data structures |
| **Broadcasting** | Efficient batch operations |
| **Gradient Vectors** | Optimization, training neural networks |

---

## Recommended Practice Problems

1. **Implement a neural network layer from scratch** using only NumPy
2. **Code PCA without sklearn** and visualize 2D projections of high-dimensional data
3. **Implement gradient descent** for linear regression
4. **Build a simple autoencoder** and visualize the learned representations
5. **Code attention mechanism** from scratch
6. **Implement batch normalization** using matrix operations
7. **Create a recommendation system** using SVD
8. **Solve least squares problems** for polynomial regression
9. **Implement cosine similarity** for text document comparison
10. **Build a simple RNN cell** using matrix operations

---

## Further Resources

**Books:**
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Linear Algebra and Its Applications" by Gilbert Strang
- "Matrix Computations" by Golub and Van Loan

**Online Courses:**
- MIT 18.06: Linear Algebra (Gilbert Strang)
- Stanford CS229: Machine Learning (Andrew Ng)
- Fast.ai: Computational Linear Algebra

**Libraries:**
- NumPy: Fundamental array operations
- SciPy: Advanced scientific computing
- PyTorch/TensorFlow: Deep learning frameworks
- scikit-learn: Machine learning algorithms

---

**End of Tutorial**

This tutorial covers the essential linear algebra concepts needed for understanding and implementing AI and machine learning algorithms. Practice implementing these concepts in code to build strong intuition!
