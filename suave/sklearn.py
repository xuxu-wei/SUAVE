"""
Scikit-learn compatible wrapper for SUAVE 
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import torch.nn.functional as F

from .utils import *
from .suave import SUAVE

class SuaveClassifier(SUAVE, BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Scikit-learn compatible wrapper for the Hybrid VAE and Multi-Task Predictor.

    This class extends the `SUAVE` by adding methods compatible with scikit-learn's API,
    such as `fit`, `transform`, `predict`, and `score`.

    Methods
    -------
    fit(X, Y, *args, **kwargs)
        Fit the model to input features `X` and targets `Y`.
    transform(X)
        Transform input samples into latent space representations.
    inverse_transform(Z)
        Reconstruct samples from latent space representations.
    predict_proba(X, deterministic=True)
        Predict probabilities for each task, either deterministically (using the latent mean) or stochastically.
    predict(X, threshold=0.5)
        Predict binary classifications for each task based on a threshold.
    score(X, Y, ...)
        Compute evaluation metrics (e.g., AUC) for each task on the given dataset.
    eval_loss(X, Y)
        Compute the total loss, including reconstruction, KL divergence, and task-specific losses.
    get_feature_names_out(input_features=None)
        Get output feature names for the latent space.

    Attributes
    ----------
    feature_names_in_ : list of str
        Feature names for the input data. Automatically populated when `X` is a pandas DataFrame during `fit`.

    Notes
    -----
    - This wrapper is designed to integrate seamlessly with scikit-learn pipelines and workflows.
    - The `transform` method maps input data into the latent space, which can be used for dimensionality reduction.
    - The `predict` and `predict_proba` methods support multi-task binary classification.
    """

    def fit(self, X, y, *args, **kwargs):
        """
        see `HybridVAEMultiTaskModel.fit` 
        """
        # Record feature names if provided (e.g., pandas DataFrame)
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        return super().fit(X, y, *args, **kwargs)
    
    def transform(self, X, return_latent_sample=False):
        """
        Transforms the input samples into the latent space.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        return_latent_sample : bool, optional
            If True, returns a sampled latent representation `z` instead of the mean `mu`.
            Default is False.

        Returns
        -------
        Z : np.ndarray
            Latent space representations with shape (n_samples, latent_dim).
            If `return_latent_sample` is True, returns sampled latent vectors; otherwise, returns the mean.
        """
        # Input validation
        if not isinstance(X, (torch.Tensor, np.ndarray)):
            raise ValueError("Input X must be a torch.Tensor or numpy.ndarray.")
        if X.ndim != 2:
            raise ValueError(f"Input X must have shape (n_samples, n_features). Got shape {X.shape}.")

        X = self._check_tensor(X).to(DEVICE)
        self.eval()
        results = []

        with torch.no_grad():
            for i in range(0, X.size(0), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                mu, logvar = self.vae.encoder(X_batch)
                if return_latent_sample:
                    z = self.vae.reparameterize(mu, logvar)
                    results.append(z.cpu().numpy())
                else:
                    results.append(mu.cpu().numpy())

        return np.vstack(results)

    def sample_latent(self, X, n_samples=1):
        """
        Sample from the latent space using reparameterization trick.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        n_samples : int, optional
            Number of samples to generate for each input (default is 1).

        Returns
        -------
        Z : np.ndarray
            Sampled latent representations with shape (n_samples, latent_dim).
        """
        X = self._check_tensor(X).to(DEVICE)
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encoder(X)
            Z = [self.vae.reparameterize(mu, logvar) for _ in range(n_samples)]
        return torch.stack(Z, dim=1).cpu().numpy()  # Shape: (input_samples, n_samples, latent_dim)
    
    def inverse_transform(self, Z):
        """
        Reconstructs samples from the latent space.

        Parameters
        ----------
        Z : np.ndarray or torch.Tensor
            Latent space representations with shape (n_samples, latent_dim).

        Returns
        -------
        X_recon : np.ndarray
            Reconstructed samples with shape (n_samples, input_dim).
        """
        Z = self._check_tensor(Z).to(DEVICE)
        self.eval()
        with torch.no_grad():
            recon = self.vae.decoder(Z)
        return recon.cpu().numpy()

    def predict_proba(self, X, deterministic=True):
        """
        Predicts probabilities for each task with optional batch processing, using numpy arrays for efficiency.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        deterministic : bool, optional
            If True, uses the latent mean (mu) for predictions, avoiding randomness.
            If False, samples from the latent space using reparameterization trick.

        Returns
        -------
        probas : list of np.ndarray
            Probabilities for each task. Each element is an array of shape (n_samples, n_classes).
        """
        X = self._check_tensor(X).to(DEVICE)
        self.eval()  # Ensure model is in evaluation mode

        # Initialize numpy arrays for storing results
        n_samples = X.size(0)
        probas_per_task = [
            np.zeros((n_samples, num_classes)) for num_classes in self.predictor.task_classes
        ]

        with torch.no_grad():
            # Process data in batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                recon, mu, logvar, z, task_outputs = self(X_batch, deterministic=deterministic)

                # Convert logits to probabilities for each task and store in numpy arrays
                for t, task_output in enumerate(task_outputs):
                    probas = F.softmax(task_output, dim=1).detach().cpu().numpy()

                    # Copy batch probabilities into preallocated numpy array
                    probas_per_task[t][i:i + self.batch_size] = probas

        return probas_per_task 

    def predict(self, X, threshold=0.5):
        """
        Predicts classifications for each task, compatible with multi-class and binary classification.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        threshold : float, optional
            Decision threshold for binary classification (default is 0.5). Ignored for multi-class tasks.

        Returns
        -------
        predictions : list of np.ndarray
            Predictions for each task. Each element is an array of shape (n_samples,).
            For binary tasks, this contains {0, 1}; for multi-class tasks, it contains {0, 1, ..., n_classes-1}.
        """
        probas = self.predict_proba(X)
        predictions = []

        for t, task_probas in enumerate(probas):
            num_classes = self.predictor.task_classes[t]
            if num_classes > 2:
                # Multi-class: return class with highest probability
                predictions.append(np.argmax(task_probas, axis=1))
            else:
                # Binary: use threshold to predict class
                predictions.append((task_probas[:, 1] >= threshold).astype(int))

        return predictions  # List of arrays, one per task


    def score(self, X, Y, *args, **kwargs):
        """
        Computes AUC scores for each task.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        Y : np.ndarray
            Ground truth labels with shape (n_samples, n_tasks).

        Returns
        -------
        scores : np.ndarray
            AUC scores for each task, shape (n_tasks,).
        """
        self.eval()
        probas_per_task = self.predict_proba(X) # expcted on cpu numpy
        Y = to_numpy(Y) # ensure cpu numpy array

        # Calculate AUC for each task (detach to avoid affecting gradient)
        auc_scores = []
        for t, (task_proba, target) in enumerate(zip(probas_per_task, Y.T)):
            num_classes = self.predictor.task_classes[t]
            # Compute probabilities for AUC
            if num_classes > 2:
                # Multi-class task: apply softmax for probabilities
                task_output_prob = task_proba
            else:
                # Binary task: apply sigmoid for probabilities of positive class
                task_output_prob = task_proba[:, 1]  # Use prob of class 1
            # Calculate AUC
            try:
                auc = roc_auc_score(y_true=target, y_score=task_output_prob, multi_class='ovo', average='macro', *args, **kwargs)
            except ValueError:
                auc = 0.5  # Default AUC for invalid cases (e.g., all labels are the same)
            auc_scores.append(auc)

        return np.array(auc_scores)

    def eval_loss(self, X, Y):
        X = self._check_tensor(X).to(DEVICE)
        Y = self._check_tensor(Y).to(DEVICE)
        self.eval()
        # Forward pass
        with torch.no_grad():
            recon, mu, logvar, z, task_outputs = self(X)
            total_loss, recon_loss, kl_loss, task_loss, per_task_losses, auc_scores = self.compute_loss(
                X, recon, mu, logvar, z, task_outputs, Y, 
                beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas, normalize_loss=False
            )

        # Convert losses to NumPy arrays
        return (total_loss.item() / len(X),  # Convert scalar tensor to Python float
                recon_loss.item() / len(X),
                kl_loss.item() / len(X),
                task_loss.item() / len(X))
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (latent space).

        Returns
        -------
        output_feature_names : list of str
            Output feature names for the latent space.
        """
        return [f"latent_{i}" for i in range(self.vae.encoder.latent_mu.out_features)]
