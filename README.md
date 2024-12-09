# Hybrid Variational Autoencoder (VAE) with Multi-Task Learning (MTL)

This repository provides an implementation of a **Hybrid Variational Autoencoder (VAE)** integrated with **Multi-Task Learning (MTL)** for dimensionality reduction and predictive modeling. The model uniquely combines **unsupervised latent representation learning** with **supervised learning**, allowing the latent space to be guided by label information for downstream prediction tasks. 

---

## Key Features

### 1. Hybrid Learning Paradigm
- **Unsupervised Learning**: The VAE learns a latent space representation by reconstructing input features and regularizing the latent space via KL divergence.  
- **Supervised Learning**: Label information directs the latent representation to align with predictive tasks (e.g., binary classification).  

This dual-objective framework ensures that the latent space is both informative and discriminative, making it well-suited for tasks requiring both dimensionality reduction and accurate predictions.

### 2. Multi-Task Learning
- **Parallel Tasks**: Supports multiple binary classification tasks simultaneously (e.g., predicting in-hospital and 28-day mortality).  
- **Task-Specific Networks**: Task-specific heads are built on shared latent representations, leveraging shared features while fine-tuning for individual outcomes.  

### 3. Flexible Architecture
- **Customizable Networks**: Dynamic encoder-decoder design supports variable depth, layer scaling (constant, linear, or geometric), and regularization options.  
- **Batch Normalization and Dropout**: Improves training stability and prevents overfitting.  

### 4. Scikit-Learn Integration
- The `HybridVAEMultiTaskSklearn` wrapper is  compatible with scikit-learn pipelines, enabling easy integration with tools like `GridSearchCV` for hyperparameter tuning.

### 5. Training Utilities
- **Joint Loss Optimization**: Combines VAE losses (reconstruction + KL divergence) with task-specific binary cross-entropy loss.  
- **Early Stopping**: Prevents overfitting by monitoring validation losses.  
- **Learning Rate Scheduling**: Automatically adjusts learning rates based on validation performance.  

---

## Example Applications

- **Dimensionality Reduction with Supervision**: Use the VAE to extract low-dimensional features that are informed by label relationships.  
- **Multi-Task Classification**: Predict multiple related outcomes (e.g., survival rates across different time horizons) with shared feature learning.  
- **Generative Modeling**: Generate synthetic samples or interpolate between data points in the latent space.  

---

## Installation

Clone this repository and install the required dependencies:
```bash
git clone https://github.com/xuxu-wei/HybridVAE.git
```

---

## Quick Start

### Define and Train the Model
```python
from HybridVAE import HybridVAEMultiTaskSklearn

# Initialize the model
model = HybridVAEMultiTaskSklearn(input_dim=X_train.shape[1],          # Input feature dimension
                                  task_count=Y_train.shape[1],         # Number of binary classification tasks
                                  latent_dim=10,                       # Latent space dimension
                                  )

# Train the model
model.fit(X_train, Y_train, epochs=1000, animate_monitor=True, verbose=1)

# Evaluate performance
predictions = model.predict(X_test)
auc_scores = model.score(X_test, Y_test)
print("AUC Scores:", auc_scores)
```

### Transform Features to Latent Space
```python
# Obtain latent representations
latent_features = model.transform(X_test)

# Reconstruct input from latent space
reconstructed_features = model.inverse_transform(latent_features)
```



## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request to suggest improvements, report bugs, or add features.



## License

This project is licensed under the **BSD 3-Clause License** . See the `LICENSE` file for details.


