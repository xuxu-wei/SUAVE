"""
Scikit-learn compatible wrapper for SUAVE.

SUAVE combines a Variational Autoencoder (VAE) for dimensionality reduction
with a Multi-Task Predictor for performing parallel predictive tasks.
This class extends the `SUAVE` by adding methods compatible with scikit-learn's API,
such as `fit`, `transform`, `predict`, and `score`.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import torch.nn.functional as F

from .utils import *
from .suave import SUAVE

class SuaveClassifier(SUAVE, BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Scikit-learn compatible wrapper for SUAVE.

    SUAVE integrates a Variational Autoencoder (VAE) for dimensionality reduction
    with a Multi-Task Predictor to handle multiple parallel predictive tasks.
    This wrapper facilitates seamless integration with scikit-learn pipelines and workflows.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.

    task_count : int
        Number of parallel prediction tasks.

    layer_strategy : str, optional, default='linear'
        Strategy for scaling hidden layer dimensions in both the VAE and Multi-Task Predictor.
        - 'constant' or 'c': All hidden layers have the same width.
        - 'linear' or 'l': Linearly increase/decrease the width.
        - 'geometric' or 'g': Geometrically increase/decrease the width.

    vae_hidden_dim : int, optional, default=64
        Number of neurons in the first hidden layer of the VAE encoder/decoder.

    vae_depth : int, optional, default=1
        Number of hidden layers in the VAE encoder/decoder.

    vae_dropout_rate : float, optional, default=0.3
        Dropout rate for VAE hidden layers.

    latent_dim : int, optional, default=10
        Dimension of the latent space in the VAE.

    predictor_hidden_dim : int, optional, default=64
        Number of neurons in the first hidden layer of the Multi-Task Predictor.

    predictor_depth : int, optional, default=1
        Number of shared hidden layers in the Multi-Task Predictor.

    predictor_dropout_rate : float, optional, default=0.3
        Dropout rate for Multi-Task Predictor hidden layers.

    vae_lr : float, optional, default=1e-3
        Learning rate for the VAE optimizer.

    vae_weight_decay : float, optional, default=1e-3
        Weight decay (L2 regularization) for the VAE optimizer.

    multitask_lr : float, optional, default=1e-3
        Learning rate for the Multi-Task Predictor optimizer.

    multitask_weight_decay : float, optional, default=1e-3
        Weight decay (L2 regularization) for the Multi-Task Predictor optimizer.

    alphas : list or torch.Tensor, optional, default=None
        Per-task weights for the task loss term, shape `(num_tasks,)`.
        If `None`, uniform weights (1 for all tasks) are used.

    beta : float, optional, default=1.0
        Weight of the KL divergence term in the VAE loss.

    gamma_task : float, optional, default=1.0
        Weight of the task loss term in the total loss.

    batch_size : int, optional, default=200
        Batch size for training.

    validation_split : float, optional, default=0.3
        Fraction of the data to use for validation.

    use_lr_scheduler : bool, optional, default=True
        Whether to enable learning rate schedulers for both the VAE and Multi-Task Predictor.

    lr_scheduler_factor : float, optional, default=0.1
        Factor by which the learning rate is reduced when the scheduler is triggered.

    lr_scheduler_patience : int, optional, default=50
        Number of epochs to wait for validation loss improvement before triggering the scheduler.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to hidden layers in both the VAE and Multi-Task Predictor.

    Attributes
    ----------
    feature_names_in_ : list of str
        Feature names for the input data. Automatically populated when `X` is a pandas DataFrame during `fit`.

    Methods
    -------
    fit(X, y, *args, **kwargs)
        Fit the model to input features `X` and targets `y`.

    transform(X, return_latent_sample=False)
        Transform input samples into latent space representations.

    inverse_transform(Z)
        Reconstruct samples from latent space representations.

    predict_proba(X, deterministic=True)
        Predict probabilities for each task, either deterministically (using the latent mean) or stochastically.

    predict(X, threshold=0.5)
        Predict binary classifications for each task based on a threshold.

    score(X, y, *args, **kwargs)
        Compute evaluation metrics (e.g., AUC) for each task on the given dataset.

    eval_loss(X, y)
        Compute the total loss, including reconstruction, KL divergence, and task-specific losses.

    get_feature_names_out(input_features=None)
        Get output feature names for the latent space.

    Notes
    -----
    - This wrapper is designed to integrate seamlessly with scikit-learn pipelines and workflows.
    - The `transform` method maps input data into the latent space, which can be used for dimensionality reduction.
    - The `predict` and `predict_proba` methods support multi-task binary and multi-class classification.
    """

    
    def fit(self, X, y, *args, **kwargs):
        """
        Fit the SUAVE model to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples, n_tasks)
            Target values for each task.

        *args : 
            Additional positional arguments.

        **kwargs : 
            Additional keyword arguments.

        Returns
        -------
        self : SuaveClassifier
            Fitted estimator.
        """
        # Record feature names if provided (e.g., pandas DataFrame)
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        return super().fit(X, y, *args, **kwargs)
    
    def transform(self, X, return_latent_sample=False):
        """
        Transform the input samples into the latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to transform.

        return_latent_sample : bool, default=False
            If `True`, returns a sampled latent representation `z` instead of the mean `mu`.
            This introduces stochasticity into the transformation.

        Returns
        -------
        Z : ndarray of shape (n_samples, latent_dim)
            Latent space representations.
            - If `return_latent_sample` is `False`, returns the mean `mu`.
            - If `True`, returns sampled latent vectors `z`.
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
        Sample from the latent space using the reparameterization trick.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to encode.

        n_samples : int, default=1
            Number of latent samples to generate for each input sample.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_samples, latent_dim)
            Sampled latent representations.
        """
        X = self._check_tensor(X).to(DEVICE)
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encoder(X)
            Z = [self.vae.reparameterize(mu, logvar) for _ in range(n_samples)]
        return torch.stack(Z, dim=1).cpu().numpy()  # Shape: (input_samples, n_samples, latent_dim)
    
    def inverse_transform(self, Z):
        """
        Reconstruct samples from the latent space.

        Parameters
        ----------
        Z : array-like of shape (n_samples, latent_dim)
            Latent space representations.

        Returns
        -------
        X_recon : ndarray of shape (n_samples, n_features)
            Reconstructed input samples.
        """
        Z = self._check_tensor(Z).to(DEVICE)
        self.eval()
        with torch.no_grad():
            recon = self.vae.decoder(Z)
        return recon.cpu().numpy()

    def predict_proba(self, X, deterministic=True):
        """
        Predict probabilities for each task.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to predict.

        deterministic : bool, default=True
            If `True`, uses the latent mean (`mu`) for predictions, ensuring deterministic outputs.
            If `False`, samples from the latent space using the reparameterization trick, introducing stochasticity.

        Returns
        -------
        probas_per_task : list of ndarray
            List containing probability estimates for each task.
            Each element is an array of shape (n_samples, n_classes) for the respective task.
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
        Predict classifications for each task.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to predict.

        threshold : float, default=0.5
            Decision threshold for binary classification tasks.
            Ignored for multi-class tasks where the class with the highest probability is selected.

        Returns
        -------
        predictions : list of ndarray
            List containing predictions for each task.
            - For binary tasks: array of shape (n_samples,) with binary labels {0, 1}.
            - For multi-class tasks: array of shape (n_samples,) with class labels.
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
        Compute evaluation metrics (e.g., AUC) for each task.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples, n_tasks)
            True labels for each task.

        *args : 
            Additional positional arguments for metric computation.

        **kwargs : 
            Additional keyword arguments for metric computation.

        Returns
        -------
        scores : ndarray of shape (n_tasks,)
            Evaluation scores for each task.
            For example, AUC scores if `roc_auc_score` is used.
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
        """
        Compute the total loss, including reconstruction, KL divergence, and task-specific losses.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples, n_tasks)
            True labels for each task.

        Returns
        -------
        total_loss : float
            Total loss averaged over the samples.

        recon_loss : float
            Reconstruction loss averaged over the samples.

        kl_loss : float
            KL divergence loss averaged over the samples.

        task_loss : float
            Task-specific loss averaged over the samples.
        """
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
        Get output feature names for the latent space.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Input feature names. If `None`, default names are generated.

        Returns
        -------
        output_feature_names : list of str
            Output feature names for the latent space.
        """
        return [f"latent_{i}" for i in range(self.vae.encoder.latent_mu.out_features)]

SuaveClassifier.fit.__doc__ = SUAVE.fit.__doc__