# SUAVE: Supervised and Unified Analysis of Variational Embeddings

**SUAVE** is a Python package built upon a **Hybrid Variational Autoencoder (VAE)** integrated with Multi-Task Learning. It unifies unsupervised latent representation learning with supervised prediction tasks. By guiding the latent space with label information, SUAVE not only achieves dimensionality reduction but also yields discriminative and interpretable embeddings that directly benefit downstream classification or regression tasks.

---

## Key Features

### 1. Supervised & Unsupervised Fusion
- **Unsupervised (VAE)**: Learns a latent space representation by reconstructing input features and regularizing the latent variables using a Kullback-Leibler (KL) divergence term.  
- **Supervised (MTL)**: Incorporates label information to shape the latent space, ensuring that the learned features are informative for one or multiple prediction tasks.

### 2. Multi-Task Learning Integration
- **Shared Representations**: A single latent space underpins multiple related classification (or other) tasks, leveraging common data structure for efficient, joint learning.  
- **Task-Specific Heads**: Independent prediction heads are built atop the shared latent space. This encourages knowledge transfer among tasks and can improve predictive performance on each one.

### 3. Flexible and Customizable Architecture
- **Configurable Networks**: Easily adjust encoder and decoder depths, widths, and layer scaling strategies (e.g., constant, linear, geometric).  
- **Regularization Built-In**: Batch normalization and dropout help stabilize training and mitigate overfitting.

### 4. Scikit-Learn Compatibility
- **Seamless Integration**: The `SuaveSklearn` class is compatible with scikit-learn’s pipeline and model selection APIs. Perform hyperparameter tuning with `GridSearchCV` and integrate SUAVE models into complex ML workflows with minimal friction.

### 5. Comprehensive Training Utilities
- **Joint Objective Optimization**: Simultaneously optimizes the VAE reconstruction/KL losses and supervised binary cross-entropy losses.  
- **Early Stopping & LR Scheduling**: Monitors validation metrics for early stopping and dynamically adjusts learning rates to ensure stable convergence.

---

## Example Use Cases

- **Supervised Dimensionality Reduction**: Obtain a low-dimensional feature representation that preserves predictive signals for classification tasks.  
- **Multi-Task Classification**: Tackle multiple related outcomes (e.g., multiple mortality endpoints) within a unified model and benefit from shared latent factors.  
- **Generative Modeling & Data Insight**: Interpolate, generate synthetic samples, and visualize latent structures that capture underlying data patterns and decision boundaries.

---

## Installation

**Please Note** This package requires PyTorch. Please install the appropriate version of PyTorch for your system using the [official PyTorch guide](https://pytorch.org/get-started/locally/).

Clone this repository and install the required dependencies:
```bash
git clone https://github.com/xuxu-wei/SUACVE.git

cd SUAVE

pip install -r requirements.txt
```

---

## Quick Start

### Define and Train the Model
```python
from SUAVE import SuaveClassifier

# Instantiate the model
model = SuaveClassifier(
    input_dim=X_train.shape[1],    # Input feature dimension
    task_count=Y_train.shape[1],   # Number of binary classification tasks
    latent_dim=10                  # Latent dimension
)

# Fit the model on training data
model.fit(X_train, Y_train, epochs=1000, animate_monitor=True, verbose=1)

# Generate predictions on test data
predictions = model.predict_proba(X_test)
auc_scores = model.score(X_test, Y_test)
print("AUC Scores:", auc_scores)

# Extract latent representations
latent_features = model.transform(X_test)

# Reconstruct inputs from latent space
reconstructed = model.inverse_transform(latent_features)
```

### Transform Features to Latent Space
```python
# Obtain latent representations
latent_features = model.transform(X_test)

# Reconstruct input from latent space
reconstructed_features = model.inverse_transform(latent_features)
```


## License

This project is licensed under the **BSD 3-Clause License** . See the `LICENSE` file for details.


