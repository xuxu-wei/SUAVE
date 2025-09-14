import os, sys, json
import inspect
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from ._base import ResetMixin
from .utils import DEVICE, generate_hidden_dims, is_interactive_environment

# TODO 支持回归任务
# 考虑建立一个新的类，用于回归任务（因为回归损失的尺度与分类损失差异太大 很难管理 一起训练可能效果很差）
# 考虑如何尽可能复用代码 拓展 compute_loss, eval_loss， plot_loss

# TODO 自定义模型：
# 引入一个类，使得不同的下游任务predictor模块可以通过parallel的方式连接到latent space上用于预测
# （但这个模型不支持训练，如需训练应该使用前面的部分训练，完成后再把模型接到这里）

# TODO 增加可视化潜空间分析工具

class Encoder(nn.Module, ResetMixin):
    """
    Encoder network for Variational Autoencoder (VAE).

    Parameters
    ----------
    input_dim : int
        Dimension of the input data (typically same as the input feature dimension).

    depth : int, optional, default=3
        Number of hidden layers in the encoder.

    hidden_dim : int, optional, default=64
        Number of neurons in the first hidden layer.

    dropout_rate : float, optional, default=0.3
        Dropout rate for regularization in the hidden layers.

    latent_dim : int, optional, default=10
        Dimension of the latent space.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to hidden layers.

    strategy : str, optional, default="linear"
        Strategy for scaling hidden layer dimensions:
        - "constant" or "c": All hidden layers have the same width.
        - "linear" or "l": Linearly decrease the width from `hidden_dim` to `latent_dim`.
        - "geometric" or "g": Geometrically decrease the width from `hidden_dim` to `latent_dim`.

    Attributes
    ----------
    body : nn.Sequential
        Sequential container for the encoder's hidden layers.

    latent_mu : nn.Linear
        Linear layer mapping the final hidden layer to the latent space mean.

    latent_logvar : nn.Linear
        Linear layer mapping the final hidden layer to the latent space log-variance.

    Methods
    -------
    forward(x)
        Perform a forward pass through the encoder, computing the latent mean (`mu`) and log-variance (`logvar`).

    Notes
    -----
    - The hidden layer dimensions are dynamically generated using the `generate_hidden_dims` function.
    - The final layer dimensions are mapped to the latent space using `latent_mu` and `latent_logvar`.

    """
    def __init__(self, input_dim, depth=3, hidden_dim=64, dropout_rate=0.3, latent_dim=10, use_batch_norm=True, strategy="linear"):
        super(Encoder, self).__init__()

        # Generate hidden dimensions for encoder
        dims_gen = list(generate_hidden_dims(hidden_dim, latent_dim, depth, strategy=strategy, order="decreasing"))

        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_rate))

        # set default for depth=0
        hidden_in_dim = hidden_dim
        hidden_out_dim = hidden_dim
        for hidden_in_dim, hidden_out_dim in dims_gen:
            layers.append(nn.Linear(hidden_in_dim, hidden_out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        self.body = nn.Sequential(*layers)
        # Latent space: mean and log variance
        self.latent_mu = nn.Linear(hidden_out_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_out_dim, latent_dim)
        
        # Extract hidden dimensions from the body
        self.dim_list = [
            (m.in_features, m.out_features) for m in self.body if isinstance(m, nn.Linear)
        ] + [(hidden_out_dim, latent_dim)]
        
    def forward(self, x):
        '''
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_dim).

        Returns
        -------
        mu : torch.Tensor
            Latent mean tensor with shape (batch_size, latent_dim).

        logvar : torch.Tensor
            Latent log variance tensor with shape (batch_size, latent_dim).
        '''
        h = self.body(x)
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        return mu, logvar
    

class Decoder(nn.Module, ResetMixin):
    """
    Decoder network for Variational Autoencoder (VAE).

    Parameters
    ----------
    dim_list : list of tuple
        List of tuples representing the input and output dimensions for each layer.
        The list should start with the latent space dimensions and end with the output space dimensions.
        For example:
        [(latent_dim, hidden_dim_1), (hidden_dim_1, hidden_dim_2), ..., (hidden_dim_last, output_dim)]
        - `latent_dim` : Dimension of the latent space.
        - `output_dim` : Dimension of the reconstructed input space (same as the input dimension in Encoder).
        The length of `dim_list` determines the number of layers in the decoder.

    dropout_rate : float, optional, default=0.3
        Dropout rate for regularization in the hidden layers.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to hidden layers.

    Attributes
    ----------
    body : nn.Sequential
        Sequential container for the hidden layers of the decoder.

    output_layer : nn.Linear
        Linear layer mapping the final hidden layer to the reconstructed input space.

    Methods
    -------
    forward(z)
        Perform a forward pass through the decoder, reconstructing the input from the latent representation.

    Notes
    -----
    - The `dim_list` parameter allows complete flexibility in defining the decoder structure.
    - The hidden layers are created using the dimensions specified in `dim_list`, except the final tuple, 
      which is used to define `output_layer`.
    - If `use_batch_norm=True`, batch normalization is applied after each hidden layer.
    """
    def __init__(self, dim_list, dropout_rate=0.3, use_batch_norm=True):
        super(Decoder, self).__init__()
        
        # Input validation
        if not isinstance(dim_list, list) or len(dim_list) < 2:
            raise ValueError("dim_list must be a list with at least two tuples [(latent_dim, hidden_dim), ..., (hidden_dim_last, output_dim)]")
        if not all(isinstance(t, tuple) and len(t) == 2 for t in dim_list):
            raise ValueError("Each element of dim_list must be a tuple (input_dim, output_dim)")

        # Build hidden layers
        layers = []
        for d, (input_dim, output_dim) in enumerate(dim_list):
            if d < len(dim_list) - 1:  # All layers except the last one
                layers.append(nn.Linear(input_dim, output_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.body = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor with shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor with shape (batch_size, output_dim).
        """
        h = self.body(z)
        return self.output_layer(h)
    

class VAE(nn.Module, ResetMixin):
    """
    Variational Autoencoder (VAE) with Encoder and Decoder.

    Parameters
    ----------
    input_dim : int, optional, default=30
        Dimension of the input data.

    depth : int, optional, default=3
        Number of hidden layers in both the encoder and decoder.

    hidden_dim : int, optional, default=64
        Number of neurons in the first hidden layer of the encoder/decoder.

    dropout_rate : float, optional, default=0.3
        Dropout rate for regularization in the encoder/decoder hidden layers.

    latent_dim : int, optional, default=10
        Dimension of the latent space.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to the encoder/decoder hidden layers.

    strategy : str, optional, default='linear'
        Strategy for scaling hidden layer dimensions in the encoder/decoder:
        - "constant" or "c": All hidden layers have the same width.
        - "linear" or "l": Linearly increase/decrease the width.
        - "geometric" or "g": Geometrically increase/decrease the width.

    Attributes
    ----------
    encoder : Encoder
        The encoder network for mapping input data to latent space.

    decoder : Decoder
        The decoder network for reconstructing input data from latent space.

    Methods
    -------
    forward(x, deterministic=False)
        Perform a forward pass through the VAE:
        1. Encode the input to obtain `mu` and `logvar` (latent mean and log-variance).
        2. Reparameterize to sample latent vectors.
        3. Decode the sampled latent vectors to reconstruct the input.

    reparameterize(mu, logvar, deterministic=False)
        Apply the reparameterization trick to sample latent vectors from `mu` and `logvar`.

    Notes
    -----
    - The encoder and decoder share the same `depth`, `hidden_dim`, `dropout_rate`, `use_batch_norm`, and `strategy` parameters.
    - The latent space sampling uses the reparameterization trick: `z = mu + std * eps`, where `std = exp(0.5 * logvar)` and `eps ~ N(0, I)`.
    - The reconstructed output has the same dimension as the input.
    """
    def __init__(self, input_dim=30, depth=3, hidden_dim=64, dropout_rate=0.3, latent_dim=10, use_batch_norm=True, strategy='linear'):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, depth, hidden_dim, dropout_rate, latent_dim, use_batch_norm, strategy=strategy)
        dim_list = [(out_dim, in_dim) for (in_dim, out_dim) in self.encoder.dim_list[::-1]] # inverse encoder layer structure
        self.decoder = Decoder(dim_list, dropout_rate, use_batch_norm)
    
    @staticmethod
    def reparameterize(mu, logvar, deterministic=False):
        """
        Reparameterization trick to sample latent representation.

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean tensor.

        logvar : torch.Tensor
            Latent log variance tensor.

        deterministic : bool, optional, default=False
            If True, uses the latent mean (`mu`) for predictions, avoiding randomness.
            If False, samples from the latent space using reparameterization trick.

        Returns
        -------
        torch.Tensor
            Sampled latent tensor with shape (batch_size, latent_dim).

        """
        if deterministic:
            # for deterministic prediction and validation
            return mu
        else:
            # Reparameterization trick: z = mu + std * eps
            # where std = exp(0.5 * logvar) and eps ~ N(0, I)
            std = torch.exp(0.5 * logvar)  # Compute standard deviation from log variance
            eps = torch.randn_like(std)   # Sample standard Gaussian noise
            return mu + eps * std         # Reparameterize to compute latent vector z

    def forward(self, x, deterministic=False):
        """
        Perform a forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_dim).

        deterministic : bool, optional, default=False
            If True, uses the latent mean (`mu`) for predictions, avoiding randomness.
            If False, samples from the latent space using reparameterization trick.

        Returns
        -------
        recon : torch.Tensor
            Reconstructed input tensor with shape (batch_size, input_dim).

        mu : torch.Tensor
            Latent mean tensor with shape (batch_size, latent_dim).

        logvar : torch.Tensor
            Latent log variance tensor with shape (batch_size, latent_dim).

        z : torch.Tensor
            Latent representation tensor with shape (batch_size, latent_dim).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon = self.decoder(z)
        return recon, mu, logvar, z

class ResidualBlock(nn.Module):
    """
    Residual Block with optional projection and batch normalization.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.

    output_dim : int
        Dimension of the output features.

    dropout_rate : float, optional, default=0.3
        Dropout rate for regularization in the hidden layers.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to hidden layers.

    Attributes
    ----------
    fc : nn.Linear
        Linear transformation layer.

    use_projection : bool
        Indicates whether a projection layer is used to match input and output dimensions.

    projection : nn.Linear or nn.Identity
        Projection layer to match input and output dimensions if necessary.

    bn : nn.BatchNorm1d or nn.Identity
        Batch normalization layer applied after the linear transformation.

    dropout : nn.Dropout
        Dropout layer for regularization.

    activation : nn.LeakyReLU
        Activation function applied after batch normalization and dropout.

    Methods
    -------
    forward(x)
        Perform a forward pass through the residual block.
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.use_projection = input_dim != output_dim
        self.projection = nn.Linear(input_dim, output_dim) if self.use_projection else None
        self.bn = nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        """
        Perform a forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, output_dim).
        """
        residual = self.projection(x) if self.use_projection else x
        h = self.fc(x)
        h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        return h + residual
    
class MultiTaskPredictor(nn.Module, ResetMixin):
    """
    Multi-task prediction network supporting parallel and sequential task modeling.

    This class implements a flexible structure for handling multiple predictive tasks.
    Each task has its own specific head, and the tasks can be modeled in two strategies:
    - Parallel: All tasks share the same input representation and are predicted independently.
    - Sequential: Each task depends on the previous task's probability distribution.

    Parameters
    ----------
    task_classes : list of int
        A list containing the number of classes for each task. For binary tasks, use 2.

    latent_dim : int
        Dimension of the shared latent representation input.

    hidden_dim : int
        Number of neurons in the hidden layers of each task-specific head.

    predictor_depth : int, optional, default=3
        Number of hidden layers in each task-specific head.

    task_strategy : str, optional, default='parallel'
        Strategy for modeling tasks:
        - 'parallel': Tasks are modeled independently.
        - 'sequential': Each task depends on the previous task's output.

    dropout_rate : float, optional, default=0.3
        Dropout rate for regularization in the hidden layers.

    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to the hidden layers.

    Attributes
    ----------
    task_heads : nn.ModuleList
        A list of task-specific heads, one for each task.

    Methods
    -------
    forward(z)
        Forward pass through the network, producing predictions for all tasks.

    Notes
    -----
    - In 'sequential' mode, each task's probability distribution is concatenated to the input for the next task.
    - Task-specific heads are dynamically built based on the provided parameters.
    """
    def __init__(self, task_classes, latent_dim, hidden_dim, predictor_depth=3, task_strategy='parallel', 
                 dropout_rate=0.3, use_batch_norm=True):
        super(MultiTaskPredictor, self).__init__()
        
        self.task_classes = task_classes  # Number of classes per task
        self.task_strategy = task_strategy  # Task modeling strategy: parallel or sequential
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        input_dim = latent_dim  # Initial input dimension to task heads
        for task_index, num_classes in enumerate(task_classes):

            if task_strategy == 'sequential' and task_index > 0:
                # In sequential mode, add previous task's probability dimension
                last_output_dim = task_classes[task_index - 1]
                input_dim += last_output_dim

            self.task_heads.append(self._build_task_head(num_classes, input_dim, hidden_dim, predictor_depth, dropout_rate, use_batch_norm))

    @staticmethod
    def _build_task_head(num_classes, input_dim, hidden_dim, depth, dropout_rate, use_batch_norm):
        """
        Build a task-specific head with classification output.

        Parameters
        ----------
        num_classes : int
            Number of classes for this task.

        input_dim : int
            Input dimension of the task-specific head.

        hidden_dim : int
            Number of neurons in the hidden layers.

        depth : int
            Number of hidden layers in the task-specific head.

        dropout_rate : float
            Dropout rate for regularization.

        use_batch_norm : bool
            Whether to apply batch normalization.

        Returns
        -------
        nn.Sequential
            Task-specific head with Softmax output.
        """
        layers = []
        for d in range(depth):
            input_dim = hidden_dim if d > 0 else input_dim
            layers.append(ResidualBlock(input_dim, hidden_dim, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm))
        layers.append(nn.Linear(hidden_dim, num_classes))  # Output layer
        return nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass through shared layers and task-specific heads.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor with shape (batch_size, latent_dim).

        Returns
        -------
        list of torch.Tensor
            List of task-specific prediction tensors, each with shape (batch_size, n_classes).
        """
        if self.task_strategy == 'parallel':
            outputs = [head(z) for head in self.task_heads]

        elif self.task_strategy == 'sequential':
            # 顺序级联建模
            outputs = []
            current_input = z  # 初始输入为VAE的潜在表示
            for task_head in self.task_heads:
                # 对当前任务进行预测
                task_output = task_head(current_input)
                outputs.append(task_output)
                # 将整个任务概率分布作为附加输入
                task_prob = F.softmax(task_output, dim=1)  # 计算任务的类别概率分布
                current_input = torch.cat([current_input, task_prob], dim=1)  # 传入完整的任务概率分布
        else:
            raise ValueError(f"Unknown task_strategy: {self.task_strategy}. Must be 'parallel' or 'sequential'.")

        return outputs


class SUAVE(nn.Module, ResetMixin):
    """
    Supervised and Unified Analysis of Variational Embeddings (SUAVE).

    This model combines a Variational Autoencoder (VAE) for dimensionality reduction
    with a Multi-Task Predictor for performing parallel predictive tasks.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
        
    task_classes : list of int
        A list containing the number of classes for each task. For binary tasks, use 2.
        
    task_strategy : str, optional, default='parallel'
        Strategy for modeling tasks:
        - "parallel" or "p": Tasks are modeled independently.
        - "sequential" or "s": Each task depends on the previous task's output.
        
    layer_strategy : str, optional, default='linear'
        Strategy for scaling hidden layer dimensions in both the VAE and Multi-Task Predictor:
        - "constant" or "c": All hidden layers have the same width.
        - "linear" or "l": Linearly increase/decrease the width.
        - "geometric" or "g": Geometrically increase/decrease the width.
        
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
        Learning rate for the MultiTask Predictor optimizer.
        
    multitask_weight_decay : float, optional, default=1e-3
        Weight decay (L2 regularization) for the MultiTask Predictor optimizer.
        
    alphas : list or torch.Tensor, optional, default=None
        Per-task weights for the task loss term, shape `(num_tasks,)`. 
        If `None`, uniform weights (1 for all tasks) are used.
        
    beta : float, optional, default=1.0
        Weight of the KL divergence term in the VAE loss.
        
    gamma_task : float, optional, default=1.0
        Weight of the task loss term in the total loss.
        
    batch_size : int, optional, default=32
        Batch size for training.
        
    validation_split : float, optional, default=0.3
        Fraction of the data to use for validation.
        
    use_lr_scheduler : bool, optional, default=True
        Whether to enable learning rate schedulers for both the VAE and Multi-Task Predictor.
        
    lr_scheduler_factor : float, optional, default=0.1
        Factor by which the learning rate is reduced when the scheduler is triggered.
        
    lr_scheduler_patience : int, optional, default=None
        Number of epochs to wait for validation loss improvement before triggering the scheduler.
        If `None`, defaults to `patience * 1/3`.
        
    use_batch_norm : bool, optional, default=True
        Whether to apply batch normalization to hidden layers in both the VAE and Multi-Task Predictor.

    Attributes
    ----------
    vae : VAE
        Variational Autoencoder for dimensionality reduction.

    predictor : MultiTaskPredictor
        Multi-task prediction module for performing parallel predictive tasks.

    fit_epochs : int
        Number of training epochs.

    early_stop_method : str
        Strategy for applying early stopping to task-specific predictors after VAE early stopping.

    vae_optimizer : torch.optim.Optimizer
        Optimizer for the VAE.

    multitask_optimizer_dict : dict
        Dictionary of optimizers for each task-specific predictor.

    training_status : dict
        Dictionary tracking the training status of modules.

    Methods
    -------
    forward(x, deterministic=False)
        Forward pass through the VAE and Multi-Task Predictor.

    compute_loss(x, recon, mu, logvar, z, task_outputs, y, beta=1.0, gamma_task=1.0, alpha=None)
        Compute the total loss, combining VAE loss (reconstruction + KL divergence) and task-specific loss.

    fit(X, Y, epochs=2000, freeze_vae=False, early_stopping=True, early_stop_method=None, patience=100, verbose=True, animate_monitor=False, plot_path=None, save_weights_path=None)
        Train the model on the input data `X` and labels `Y`.

    plot_loss(train_vae_losses, val_vae_losses, train_aucs, val_aucs, train_task_losses, val_task_losses, save_path=None, display_id="loss_plot")
        Plot training and validation loss curves for VAE and task-specific losses.

    save_config(config_path)
        Save the initialization parameters to a JSON file.

    save_model(save_path, epoch)
        Save model weights.

    save_complete_model(save_dir)
        Save the complete model, including weights and configuration.

    load_config(cls, config)
        Create an instance of the model from a configuration dictionary.

    load_complete_model(cls, load_dir, device=DEVICE)
        Load a complete model, including weights and configuration.

    Notes
    -----
    - The encoder and decoder of the VAE, as well as the Multi-Task Predictor, use dynamically generated hidden layers
      based on the specified depth, strategy, and hidden dimensions.
    - Task-specific outputs in the Multi-Task Predictor use sigmoid activation for binary classification.
    """
    def __init__(self, 
                 input_dim, 
                 task_classes,
                 task_strategy='parallel', 
                 layer_strategy='linear',
                 vae_hidden_dim=64, 
                 vae_depth=1,
                 vae_dropout_rate=0.3, 
                 latent_dim=10, 
                 predictor_hidden_dim=64, 
                 predictor_depth=1,
                 predictor_dropout_rate=0.3, 
                 # training related params for param tuning
                 vae_lr=1e-3, 
                 vae_weight_decay=1e-3, 
                 multitask_lr=1e-3, 
                 multitask_weight_decay=1e-3,
                 alphas=None,
                 beta=1.0, 
                 gamma_task=1,
                 batch_size=32, 
                 validation_split=0.3, 
                 use_lr_scheduler=True,
                 lr_scheduler_factor=0.1,
                 lr_scheduler_patience=None,
                 use_batch_norm=True, 
                 ):
        super(SUAVE, self).__init__()
        self.vae = VAE(input_dim, depth=vae_depth, hidden_dim=vae_hidden_dim, strategy=layer_strategy, # VAE strcture
                       dropout_rate=vae_dropout_rate, latent_dim=latent_dim, use_batch_norm=use_batch_norm # normalization
                       )
        self.predictor = MultiTaskPredictor(task_classes=task_classes, # output_dim of each task
                                            latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, predictor_depth=predictor_depth, task_strategy=task_strategy, # task strcture
                                            dropout_rate=predictor_dropout_rate, use_batch_norm=use_batch_norm # normalization
                                            )

        self.input_dim = input_dim
        self.task_classes = task_classes
        self.task_strategy = task_strategy
        self.layer_strategy = layer_strategy
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_depth = vae_depth
        self.vae_dropout_rate = vae_dropout_rate
        self.latent_dim = latent_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.predictor_depth = predictor_depth
        self.predictor_dropout_rate = predictor_dropout_rate
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.multitask_lr = multitask_lr
        self.multitask_weight_decay = multitask_weight_decay
        self.alphas = alphas
        self.beta = beta
        self.gamma_task = gamma_task
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.use_batch_norm = use_batch_norm
        self.to(DEVICE)

        assert self.task_strategy in ['parallel', 'sequential'], f"Unknown task_strategy: {self.task_strategy}. Must be 'parallel' or 'sequential'."
        
    def forward(self, x, deterministic=False):
        """
        Forward pass through the complete model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_dim).
            
        deterministic : bool, optional, default=False
            If True, uses the latent mean (`mu`) for predictions, avoiding randomness.
            If False, samples from the latent space using reparameterization trick.

        Returns
        -------
        recon : torch.Tensor
            Reconstructed tensor with shape (batch_size, input_dim).
            
        mu : torch.Tensor
            Latent mean tensor with shape (batch_size, latent_dim).
            
        logvar : torch.Tensor
            Latent log variance tensor with shape (batch_size, latent_dim).
            
        z : torch.Tensor
            Latent representation tensor with shape (batch_size, latent_dim).
            
        task_outputs : list of torch.Tensor
            List of task-specific prediction tensors, each with shape (batch_size, n_classes).
        """
        x = self._check_tensor(x).to(DEVICE)
        recon, mu, logvar, z = self.vae(x, deterministic=deterministic)
        task_outputs = self.predictor(z)
        return recon, mu, logvar, z, task_outputs
    
    def compute_recon_loss(self, x, recon, mu, logvar):
        """
        Compute VAE reconstruction and KL divergence losses.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor, shape (batch_size, input_dim).
        
        recon : torch.Tensor
            Reconstructed input tensor, shape (batch_size, input_dim).
        
        mu : torch.Tensor
            Latent mean tensor, shape (batch_size, latent_dim).
        
        logvar : torch.Tensor
            Latent log variance tensor, shape (batch_size, latent_dim).

        Returns
        -------
        recon_loss : torch.Tensor
            Reconstruction loss value.
        
        kl_loss : torch.Tensor
            KL divergence loss value.
        """
        # Compute reconstruction loss: sum over features then mean over batch
        recon_elementwise = F.mse_loss(recon, x, reduction='none')
        recon_loss = recon_elementwise.sum(dim=1).mean()

        # KL divergence loss summed over latent dims then averaged over batch
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_per_sample.mean()

        return recon_loss, kl_loss


    def compute_pred_loss(self, task_outputs, y, alpha=None):
        """
        Compute prediction losses and AUC scores for each task.

        Parameters
        ----------
        task_outputs : list of torch.Tensor
            List of task-specific predictions, each shape (batch_size, n_classes).
        
        y : torch.Tensor
            Ground truth target tensor, shape (batch_size, num_tasks).
        
        alpha : list or torch.Tensor, optional
            Per-task weights, shape (num_tasks,). Default is uniform weights.

        Returns
        -------
        task_loss_sum : torch.Tensor
            Sum of all weighted task-specific losses.
        
        per_task_losses : list of torch.Tensor
            List of task-specific loss values.
        
        auc_scores : list of float
            List of AUC scores for each task.
        """
        # Initialize task losses
        per_task_losses = []
        
        # Handle alpha for task weighting
        if alpha is None:
            alpha = torch.ones(len(task_outputs), device=DEVICE)
        else:
            alpha = torch.tensor(alpha, device=DEVICE, dtype=torch.float32)

        # Compute losses and AUCs for each task
        auc_scores = []
        for t, (task_output, target) in enumerate(zip(task_outputs, y.T)):
            # Compute mean task loss so that each task contributes on a similar
            # scale regardless of batch size
            task_loss_fn = nn.CrossEntropyLoss(reduction='mean')
            task_loss = task_loss_fn(task_output, target.long())
            weighted_task_loss = alpha[t] * task_loss
            per_task_losses.append(weighted_task_loss)

            # Compute AUC
            num_classes = self.predictor.task_classes[t]
            if num_classes > 2:
                task_output_prob = F.softmax(task_output, dim=1).detach().cpu().numpy()
            else:
                task_output_prob = F.softmax(task_output, dim=1).detach().cpu().numpy()[:, 1]

            target_cpu = target.detach().cpu().numpy()
            try:
                auc = roc_auc_score(y_true=target_cpu, y_score=task_output_prob, 
                                  multi_class='ovo', average='macro')
            except ValueError:
                auc = 0.5
            auc_scores.append(auc)

        # Sum up weighted task losses
        task_loss_sum = sum(per_task_losses)

        return task_loss_sum, per_task_losses, auc_scores

    def compute_loss(self, x, recon, mu, logvar, z, task_outputs, y, beta=1.0, gamma_task=1.0, alpha=None):
        """
        Compute total loss combining VAE and prediction losses.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor, shape (batch_size, input_dim).
        recon : torch.Tensor
            Reconstructed input tensor, shape (batch_size, input_dim).
        mu : torch.Tensor
            Latent mean tensor, shape (batch_size, latent_dim).
        logvar : torch.Tensor
            Latent log variance tensor, shape (batch_size, latent_dim).
        z : torch.Tensor
            Latent representation tensor, shape (batch_size, latent_dim).
        task_outputs : list of torch.Tensor
            List of task-specific predictions.
        y : torch.Tensor
            Ground truth target tensor.
        beta : float, optional, default=1.0
            Weight of KL divergence loss.
        gamma_task : float, optional, default=1.0
            Weight of task loss.
        alpha : list or torch.Tensor, optional
            Per-task weights.

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss value.
        recon_loss : torch.Tensor
            Reconstruction loss value.
        kl_loss : torch.Tensor
            KL divergence loss value.
        task_loss_sum : torch.Tensor
            Sum of task losses.
        per_task_losses : list of torch.Tensor
            Individual task losses.
        auc_scores : list of float
            AUC scores for each task.
        """
        # Compute VAE losses
        recon_loss, kl_loss = self.compute_recon_loss(x, recon, mu, logvar)
        
        # Compute prediction losses
        task_loss_sum, per_task_losses, auc_scores = self.compute_pred_loss(task_outputs, y, alpha)

        # Combine losses with weights
        total_loss = recon_loss + beta * kl_loss + gamma_task * task_loss_sum

        return total_loss, recon_loss, kl_loss, task_loss_sum, per_task_losses, auc_scores

    def fit(self, X, Y, 
            epochs=1000, 
            freeze_vae = False,
            predictor_fine_tuning = False,
            early_stopping=True, 
            early_stop_method=None,
            patience=50,
            verbose=True, 
            animate_monitor=False,
            plot_path=None,
            save_weights_path=None):
        """
        Train the SUAVE model on provided data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Feature matrix of shape (n_samples, n_features).
            
        Y : np.ndarray or torch.Tensor
            Target matrix of shape (n_samples, n_tasks).
            
        epochs : int, optional, default=1000
            Number of training epochs.
            
        freeze_vae : bool, optional, default=False
            - If True, VAE parameters are frozen (no gradients or optimizer steps).
            - Use this when the latent representation is already learned and stable,
              and you only want to optimize task-specific heads.
            
        predictor_fine_tuning : bool, optional, default=False
            Controls whether the predictor weights are reset during training.
            - If True, the predictor weights are not reset, allowing for fine-tuning.
            - If False, the predictor weights are reset to their initial state.
    
        early_stopping : bool, optional, default=True
            If True, enables early stopping based on validation loss improvements. Early stopping 
            is applied first to the VAE module. Once the VAE stops improving, it is effectively 'frozen'
            (learning rate set to 0), and training can continue on the predictor modules until their 
            respective early stopping criteria are met.
            
        early_stop_method : str, optional, default=None
            Strategy for applying early stopping to task-specific predictors after VAE early stopping.
            Must be either `'parallel'` or `'sequential'`. If None, defaults to the value of `task_strategy`.
            - `'parallel'`: All tasks are monitored and stopped independently based on their own validation loss.
            - `'sequential'`: Tasks are monitored and stopped in sequence, where each task is only monitored
                after the preceding tasks have been early stopped (applicable only if `task_strategy` is `'sequential'`).
                
        patience : int, optional, default=50
            Number of epochs to wait for improvement before triggering early stopping for each module.
            
        verbose : bool, optional, default=True
            If True, displays a tqdm progress bar and training metrics per epoch.
            
        animate_monitor : bool, optional, default=False
            If True and running in an interactive environment, attempts to dynamically update training 
            plots at specified intervals. If False or not in such an environment, no dynamic plotting 
            will occur.
            
        plot_path : str or None, optional, default=None
            Directory to save loss plots periodically (e.g., every few epochs). If None, tries dynamic 
            plotting if `animate_monitor` is True. Otherwise, no plot is saved.
            
        save_weights_path : str or None, optional, default=None
            Directory to save model weights periodically and at the end of training. If None, weights 
            are not saved.

        Returns
        -------
        self : SUAVE
            The fitted SUAVE model instance.

        Notes
        -----
        - **Early Stopping Strategy**: Early stopping is conducted in stages. Initially, the VAE portion is monitored. Once it fails 
          to improve beyond the patience threshold, the VAE's learning rate is set to zero. Subsequently, based on the 
          `early_stop_method`, the predictor heads are monitored either in parallel or sequentially.
          
        - **Predictor Fine-Tuning**: When `freeze_vae=True`, only the predictor modules are trained. 
          This is useful when the latent space has already been well-learned. 
          Additionally, `predictor_fine_tuning=True` ensures predictor weights are not reset.
          
        - **Early Stop Method vs. Task Strategy**: The `early_stop_method` allows independent control over how task-specific 
          predictors are stopped after VAE early stopping. It can be set to `'parallel'` or `'sequential'` regardless of the 
          `task_strategy`, providing additional flexibility.

        Examples
        --------
        1. **Training from scratch (VAE + Predictors)**:
        >>> suave_model = SUAVE(input_dim=30, task_classes=[2, 2])
        >>> suave_model.fit(X, Y, epochs=500, patience=50, early_stopping=True, animate_monitor=False)

        In this scenario, both the VAE and predictors are trained together from the beginning.

        2. **Fine-tuning the predictors**:
        Suppose you already have a trained model or a stable latent space representation. You can then 
        increase the predictor learning rate and switch to fine-tuning mode:

        >>> suave_model.set_params(multitask_lr=1e-2, lr_scheduler_factor=0.1, batch_size=32)
        >>> suave_model.fit(X, Y, patience=100, freeze_vae=True, predictor_fine_tuning=True)
        """
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae.reset_parameters()
            
        if predictor_fine_tuning:
            pass
        else:
            self.predictor.reset_parameters()
            
        self.early_stop_method = self.task_strategy if early_stop_method is None else early_stop_method
        assert self.early_stop_method in ['parallel', 'sequential'], f"Unrecognized param `early_stop_method`={early_stop_method}. Must be \'parallel\' or \'sequential\'"
        
        self.fit_epochs = epochs
        
        self._check_prediction_task_name(Y)
        # Data checks and device transfer
        X = self._check_tensor(X).to(DEVICE)
        Y = self._check_tensor(Y).to(DEVICE)
        self.to(DEVICE)  # Ensure the model is on the correct device
        
        # Data validation
        if len(X) != len(Y):
            raise ValueError("Features and targets must have the same number of samples.")
        
        if torch.isnan(X).any():
            raise ValueError("Features (X) contain NaNs.")
        
        if torch.isnan(Y).any():
            raise ValueError("Targets (Y) contain NaNs.")
        
        # Attempt to split the data multiple times
        success = False
        for attempt in range(10):  # Maximum 10 attempts to find a valid split
            # Split data into training and validation sets
            n_samples = len(X)
            n_val = int(n_samples * self.validation_split)
            perm = torch.randperm(n_samples)
            train_idx, val_idx = perm[:-n_val], perm[-n_val:]
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            if self._validate_split(Y.shape[1], Y_train, Y_val):
                success = True
                # Print class distribution for each task in validation set
                if verbose > 1:
                    print("Validation set class distribution:")
                    for task_idx in range(Y.shape[1]):
                        print(f"\nTask {self.task_names[task_idx]}:")
                        unique_classes, class_counts = torch.unique(Y_val[:, task_idx], return_counts=True)
                        for cls, count in zip(unique_classes, class_counts):
                            print(f"  Class {int(cls)}: {int(count)} samples")
                break
        if not success:
            raise ValueError("Failed to split data: One or more tasks have imbalanced classes in training or validation sets.")
        
        # Separate optimizers for VAE and MultiTask predictor
        if not freeze_vae:
            self.vae_optimizer = torch.optim.Adam(
                self.vae.parameters(),
                lr=self.vae_lr,
                weight_decay=self.vae_weight_decay,
                eps=1e-8,
            )
        else:
            self.vae_optimizer = None

        self.multitask_optimizer_dict = {
            task_name: torch.optim.Adam(
                self.predictor.task_heads[t].parameters(),
                lr=self.multitask_lr,
                weight_decay=self.multitask_weight_decay,
                eps=1e-8,
            )
            for t, task_name in enumerate(self.task_names)
        }
        
        # 初始化调度器（仅当启用时）
        if self.use_lr_scheduler:
            if self.lr_scheduler_patience is None:
                self.lr_scheduler_patience = patience * 1/3
            if not freeze_vae:
                vae_scheduler = ReduceLROnPlateau(
                    self.vae_optimizer,
                    mode='min',
                    factor=self.lr_scheduler_factor,
                    patience=self.lr_scheduler_patience,
                )
            multitask_scheduler_dict = {
                task_name: ReduceLROnPlateau(
                    self.multitask_optimizer_dict[task_name],
                    mode='min',
                    factor=self.lr_scheduler_factor,
                    patience=self.lr_scheduler_patience,
                )
                for task_name in self.task_names
            }

        # Initialize best losses and patience counters
        best_vae_loss = float('inf')
        best_per_task_losses = [float('inf')] * len(self.task_classes)
        self.vae_patience_counter = patience + 1 if freeze_vae else 0
        self.task_patience_counters = {task_name: 0 for task_name in self.task_names}  # Initialize task-specific patience counters
        self.training_status = {task_name: True for task_name in self.task_names}
        self.training_status[self._vae_task_name] = 1 - bool(freeze_vae)

        # Training and validation loss storage
        train_vae_losses, train_task_losses, train_aucs = [], [], []
        val_vae_losses, val_task_losses, val_aucs = [], [], []
        
        # Training loop with tqdm
        iterator = range(epochs)
        plot_interval = max(1, int(self.fit_epochs * 0.01))
        if verbose:  # 控制进度条显示
            if animate_monitor:
                iterator = tqdm(iterator, desc="Training", unit="epoch")
            elif is_interactive_environment():
                iterator = tqdm_notebook(iterator, desc="Training", unit="epoch")

        for epoch in iterator:
            self.train()
            train_vae_loss = 0.0
            train_task_loss_sum = 0.0
            train_per_task_losses = np.zeros(len(self.task_classes)) # 每个 task 的 loss 
            train_auc_scores_tasks = []
            train_batch_count = 0
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                Y_batch = Y_train[i:i + self.batch_size]

                # Combine with the remaining samples if the batch is too small
                if len(X_batch) <= 2:
                    if i + self.batch_size < len(X_train):  # 如果有更多的样本，合并到下一个批次
                        X_batch = torch.cat([X_batch, X_train[i + self.batch_size:i + 2 * self.batch_size]])
                        Y_batch = torch.cat([Y_batch, Y_train[i + self.batch_size:i + 2 * self.batch_size]])
                    else:
                        # 如果没有更多样本，将这批数据直接使用
                        pass

                # Ensure the batch size is still valid
                if len(X_batch) <= 2:
                    continue  # 仍然太小，跳过
                
                # Reset gradients
                if not freeze_vae:
                    self.vae_optimizer.zero_grad()
                for task_name, task_optimizer in self.multitask_optimizer_dict.items():
                    task_optimizer.zero_grad()

                # Forward pass
                recon, mu, logvar, z, task_outputs = self(X_batch)

                # Compute loss
                total_loss, recon_loss, kl_loss, task_loss_sum, per_task_losses, auc_scores = self.compute_loss(
                    X_batch, recon, mu, logvar, z, task_outputs, Y_batch, 
                    beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas
                )

                # Backward pass and optimization
                total_loss.backward()
                if not freeze_vae:
                    self.vae_optimizer.step()
                for task_name, task_optimizer in self.multitask_optimizer_dict.items():
                    task_optimizer.step()

                # Accumulate losses
                train_vae_loss += ((recon_loss.item() + self.beta * kl_loss.item()) / self.batch_size) # record batch normalized loss after backward pass
                train_task_loss_sum += (task_loss_sum.item() / self.batch_size)
                train_per_task_losses += np.array([(loss.cpu().detach().numpy() / self.batch_size) for loss in per_task_losses])
                train_auc_scores_tasks.append(np.array(auc_scores))
                train_batch_count += 1

            # Normalize training losses by the number of batches
            train_vae_loss /= train_batch_count
            train_task_loss_sum /= train_batch_count
            train_auc_scores_tasks= np.mean(np.stack(train_auc_scores_tasks), axis=0)
            
            train_vae_losses.append(train_vae_loss)
            train_task_losses.append(train_task_loss_sum)
            train_aucs.append(train_auc_scores_tasks)

            # Validation phase
            self.eval()
            val_vae_loss = 0.0
            val_task_loss_sum = 0.0 # 总 task loss
            val_per_task_losses = np.zeros(len(self.task_classes)) # 每个 task 的 loss 
            val_auc_scores_tasks = []
            val_batch_count = 0
            with torch.no_grad():
                for i in range(0, len(X_val), self.batch_size):
                    X_batch = X_val[i:i + self.batch_size]
                    Y_batch = Y_val[i:i + self.batch_size]

                    if len(X_batch) <= 2:  # Skip very small batches
                        continue

                    # Forward pass
                    recon, mu, logvar, z, task_outputs = self(X_batch)

                    # Compute validation losses
                    total_loss, recon_loss, kl_loss, task_loss_sum, per_task_losses, auc_scores = self.compute_loss(
                        X_batch, recon, mu, logvar, z, task_outputs, Y_batch, 
                        beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas
                    )

                    # Accumulate losses
                    val_vae_loss += ((recon_loss.item() + self.beta * kl_loss.item()) / self.batch_size)
                    val_task_loss_sum += (task_loss_sum.item() / self.batch_size)
                    val_per_task_losses += np.array([(loss.cpu().detach().numpy() / self.batch_size) for loss in per_task_losses])
                    val_auc_scores_tasks.append(np.array(auc_scores))
                    val_batch_count += 1

            if self.use_lr_scheduler:
                if self.training_status[self._vae_task_name] and not freeze_vae:
                    vae_scheduler.step(val_vae_loss)  # 调用时需传入验证损失

                for (task_name, task_scheduler), task_loss in zip(multitask_scheduler_dict.items(), val_per_task_losses):
                    if self.training_status[task_name]:
                        task_scheduler.step(task_loss)
                    
            # Normalize Validation losses by the number of batches
            val_vae_loss /= val_batch_count
            val_task_loss_sum /= val_batch_count
            val_auc_scores_tasks = np.mean(np.stack(val_auc_scores_tasks), axis=0)
            
            val_vae_losses.append(val_vae_loss)
            val_task_losses.append(val_task_loss_sum)
            val_aucs.append(val_auc_scores_tasks)

            # Update progress bar
            if verbose:
                train_auc_formated = [round(auc, 3) for auc in train_auc_scores_tasks]
                val_auc_formated = [round(auc, 3) for auc in val_auc_scores_tasks]
                iterator.set_postfix({
                    "VAE(t)": f"{train_vae_loss:.3f}", # train total VAE loss, normalized by batch size
                    "VAE(v)": f"{val_vae_loss:.3f}", # validation total VAE loss, normalized by batch size
                    "AUC(t)": f"{train_auc_formated}", # train AUC for each task
                    "AUC(v)": f"{val_auc_formated}" # validation AUC for each task
                })
            
            # Early stopping logic
            if early_stopping:
                if (val_vae_loss < best_vae_loss):
                    best_vae_loss = val_vae_loss
                    if not freeze_vae:
                        self.vae_patience_counter = 0
                else:
                    self.vae_patience_counter += 1

            # freeze weights if counter exceed patience
            if (not freeze_vae) and (self.vae_patience_counter >= patience):
                self.stop_training_module(self.vae, self._vae_task_name, verbose, epoch)

                # For prediction tasks, start counting for early stopping after VAE has stopped.
                if (self.early_stop_method == 'parallel') or (self.task_strategy == 'parallel'): # parallel tasks should not stop squentially
                    for t in range(len(self.task_classes)):
                        self._update_task_patience(t, val_per_task_losses, best_per_task_losses)
                
                if self.task_strategy == 'sequential':
                    # Sequential task early stopping
                    for t in range(len(self.task_classes)):
                        # 仅在前置任务已停止时允许检查当前任务
                        if  t==0 or ((t >= 1 ) and (self.training_status[self.task_names[t-1]] == False)):
                            # 如果超出耐心计数，则停止该任务训练
                            if self.early_stop_method == 'sequential':
                                self._update_task_patience(t, val_per_task_losses, best_per_task_losses)
                            if self.task_patience_counters[self.task_names[t]] >= patience:
                                self.stop_training_module(self.predictor.task_heads[t], self.task_names[t], verbose, epoch)

                elif self.task_strategy == 'parallel':
                    for t in range(len(self.task_classes)):
                        if self.task_patience_counters[self.task_names[t]] >= patience:
                            self.stop_training_module(self.predictor.task_heads[t], self.task_names[t], verbose, epoch)

            # Stop if both counters exceed patience
            if self.vae_patience_counter >= patience and all(counter >= patience for _, counter in self.task_patience_counters.items()):
                print("Early stopping triggered due to no improvement in both VAE and task losses.") if verbose > 0 else None
                if save_weights_path:
                    self.save_model(save_weights_path, "final")
                break

            # Save loss plot every 5% epochs
            if ((epoch + 1) % plot_interval == 0) and ((is_interactive_environment() and animate_monitor) or plot_path):
                loss_plot_path = None
                if plot_path:
                    loss_plot_path = os.path.join(plot_path, f"loss_epoch.jpg")
                self.plot_loss(train_vae_losses,  # orginally reduction by sum, normalize by batch size here
                               val_vae_losses, # orginally reduction by sum, normalize by batch size here
                               train_aucs, 
                               val_aucs, 
                               train_task_losses, 
                               val_task_losses,
                               save_path=loss_plot_path
                               )
            # Save weights every 20% epochs
            if ((epoch + 1) % int((self.fit_epochs * 0.2)) == 0) and save_weights_path:
                self.save_model(save_weights_path, epoch + 1)

        # Save final weights
        if save_weights_path:
            self.save_complete_model(save_weights_path)
            print(f'最终模型参数已保存: {os.path.join(save_weights_path, f"epoch_final.pth")}') if verbose > 0 else None

        return self
    
    def _update_task_patience(self, task_ix, val_losses, best_losses):
        """
        Update the patience counter for a specific task based on validation loss.

        Parameters
        ----------
        task_ix : int
            Index of the task to update.

        val_losses : list of float
            Current epoch's validation losses for all tasks.

        best_losses : list of float
            Best observed validation losses for all tasks so far.
        """
        if val_losses[task_ix] < best_losses[task_ix]:
            best_losses[task_ix] = val_losses[task_ix]
            self.task_patience_counters[self.task_names[task_ix]] = 0
        else:
            self.task_patience_counters[self.task_names[task_ix]] += 1

    def stop_training_module(self, module, name, verbose, epoch):
        """
        Freeze a module by setting its parameters to not require gradients.

        Parameters
        ----------
        module : nn.Module
            The module to be frozen (e.g., VAE or a task head).

        name : str
            The name of the module (used for logging).

        verbose : bool
            Whether to log the stopping event.

        epoch : int
            Current epoch number for logging.
        """
        #! Old version. Can cause hard Error.
        # for param in module.parameters():
        #     param.requires_grad = False

        # if self.training_status[name]:
        #     self.training_status[name] = False
        #     if verbose:
        #         print(f'Epoch {epoch + 1}: {name} early stopping triggered.')

        # Handle VAE
        if module == self.vae:
            for param_group in self.vae_optimizer.param_groups:
                param_group['lr'] = 0

            # Update training status
            if verbose and self.training_status[self._vae_task_name]: # show info at the first time
                print(f"Epoch {epoch + 1}: VAE early stopping triggered.")
            self.training_status[self._vae_task_name] = False

        # Handle task head
        elif name in self.multitask_optimizer_dict:
            # Set task head optimizer learning rate to 0
            for param_group in self.multitask_optimizer_dict[name].param_groups:
                param_group['lr'] = 0

            # Update training status
            if verbose and self.training_status[name]: # show info at the first time
                print(f"Epoch {epoch + 1}: Task {name} early stopping triggered.")
            self.training_status[name] = False
        
    def plot_loss(self, train_vae_losses, val_vae_losses, train_aucs, val_aucs, train_task_losses, val_task_losses, save_path=None, display_id="loss_plot"):
        """
        Plot training and validation loss curves for VAE and task-specific AUCs.

        Parameters
        ----------
        train_vae_losses : list of float
            List of VAE losses (reconstruction + KL) for the training set at each epoch.

        val_vae_losses : list of float
            List of VAE losses (reconstruction + KL) for the validation set at each epoch.

        train_aucs : list of np.ndarray
            List where each element is an np.ndarray of shape [n_tasks,], containing AUC scores for each task at each epoch.

        val_aucs : list of np.ndarray
            List where each element is an np.ndarray of shape [n_tasks,], containing AUC scores for each task at each epoch.

        train_task_losses : list of float
            List of total task losses (summed over all tasks) for training at each epoch.

        val_task_losses : list of float
            List of total task losses (summed over all tasks) for validation at each epoch.

        save_path : str or None, optional, default=None
            Path to save the plot image. If None, dynamically display in a notebook.

        display_id : str, optional, default="loss_plot"
            Unique identifier for the display container to ensure updates occur in the same container.
        """
        
        # Convert train_aucs and val_aucs to task-wise epoch-level averages
        train_aucs = np.array(train_aucs)  # Shape: [n_tasks, n_epochs]
        val_aucs = np.array(val_aucs)      # Shape: [n_tasks, n_epochs]

        num_tasks = train_aucs.shape[1]  # Determine the number of tasks
        total_plots = 2 + num_tasks  # One plot for VAE losses and one for each task's AUC

        # Define the grid layout
        cols = cols = min(3, total_plots) # Maximum number of plots per row
        rows = (total_plots + cols - 1) // cols  # Compute number of rows

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))  # Dynamically set figure size
        axes = axes.flatten()  # Flatten axes for easy indexing

        # VAE Loss Plot
        stop_flag = {False:'(early stopped)', True:''}[self.training_status[self._vae_task_name]]
        ax = axes[0]
        ax.plot(train_vae_losses, label='Train VAE Loss', linestyle='-')
        ax.plot(val_vae_losses, label='Val VAE Loss', linestyle='-')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(r'Reconstruction + $\beta$KL')
        ax.legend()
        ax.set_title(fr'VAE Loss (Reconstruction + $\beta$KL) {stop_flag}')
        ax.grid()

        # Task-specific AUC Plots
        for t, task_name in enumerate(self.task_names):
            stop_flag = {False:'(early stopped)', True:''}[self.training_status[task_name]]
            ax = axes[t + 1]
            ax.plot(train_aucs[:,t], label=f'Train AUC for Task {t+1}', linestyle='-')
            ax.plot(val_aucs[:,t], label=f'Validation AUC  for Task {t+1}', linestyle='-')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('AUC')
            ax.legend()
            ax.set_title(f'AUC Score for {task_name} {stop_flag}')
            ax.grid()

        # Task Loss Plot
        ax = axes[num_tasks + 1]
        ax.plot(train_task_losses, label='Train Task Loss', linestyle='-')
        ax.plot(val_task_losses, label='Val Task Loss', linestyle='-')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross Entropy')
        ax.legend()
        ax.set_title('Task Loss (Total Cross-Entropy)')
        ax.grid()

        # Hide unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=360)

        # Check if running in notebook
        if hasattr(sys, 'ps1') or ('IPython' in sys.modules and hasattr(sys, 'argv') and sys.argv[0].endswith('notebook')):
            from IPython.display import update_display, clear_output
            clear_output(wait=True)
            update_display(plt.gcf(), display_id=display_id)
            plt.pause(0.1)
        plt.close()

    def save_config(self, config_path):
        """
        Save the initialization parameters to a JSON file.

        Parameters
        ----------
        config_path : str
            Path to save the configuration file.
        """
        # Get init parameters and their default values
        init_params = inspect.signature(self.__init__).parameters

        # Only save parameters explicitly provided during initialization
        config = {
            k: getattr(self, k)
            for k in init_params if k != "self" and hasattr(self, k)
        }

        with open(config_path, "w", encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Model configuration saved to {config_path}")
    
    def save_model(self, save_path, epoch):
        """
        Save model weights.

        Parameters
        ----------
        save_path : str
            Directory to save the weights.

        epoch : int
            Current epoch number, used for naming the file.
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pth"))

    def save_complete_model(self, save_dir):
        """
        Save the complete model, including weights and configuration.

        Parameters
        ----------
        save_dir : str
            Directory to save the model configuration and weights.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_config(os.path.join(save_dir, "Hybrid_VAE_config.json"))
        self.save_model(save_dir, "final")

    @classmethod
    def load_config(cls, config):
        """
        Create an instance of the model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the model initialization parameters.

        Returns
        -------
        SUAVE
            A new instance of the model initialized with the provided configuration.
        """
        # Get the parameter signature of __init__
        init_params = inspect.signature(cls.__init__).parameters

        # Filter config keys to match the constructor's parameters
        filtered_config = {k: v for k, v in config.items() if k in init_params}

        # Ensure all required parameters are provided
        required_params = [
            name for name, param in init_params.items()
            if param.default == inspect.Parameter.empty and name != "self"
        ]
        missing_params = [p for p in required_params if p not in filtered_config]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        return cls(**filtered_config)

    @classmethod
    def load_complete_model(cls, load_dir, device=DEVICE):
        """
        Load a complete model, including weights and configuration.

        Parameters
        ----------
        load_dir : str
            Directory containing the model configuration and weights.

        device : torch.device, optional, default=DEVICE
            Device to load the model onto.

        Returns
        -------
        SUAVE
            The reconstructed model instance with weights loaded.

        Examples
        --------
        >>> suave = SUAVE.load_complete_model("complete_model/", device=torch.device('cpu'))
        """
        # Load configuration
        config_path = os.path.join(load_dir, "Hybrid_VAE_config.json")
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        model = cls.load_config(config)
        model.load_state_dict(torch.load(os.path.join(load_dir, "epoch_final.pth"), map_location=device, weights_only=True))
        
        return model

    def _check_prediction_task_name(self, Y):
        """
        Check and set prediction task names with the passed target `Y`.

        Parameters
        ----------
        Y : pd.DataFrame, pd.Series, or np.ndarray
            Ground truth target data.
        """
        self._vae_task_name = 'vae'

        if hasattr(Y, 'columns'):
            self.task_names = list(Y.columns)

        elif isinstance(Y, pd.Series):
            if not Y.name  is None:
                self.task_names = [Y.name]
        else:
            self.task_names = [f'task {i+1}' for i in range(len(self.task_classes))]

    def _check_tensor(self, X: torch.Tensor):
        """
        Ensure the input is a tensor and move it to the correct device.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data to check and convert if necessary.

        Returns
        -------
        torch.Tensor
            Tensor moved to the specified device.
        """
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE)
    
    @staticmethod
    def _validate_split(num_tasks, Y_train, Y_val):
        """
        Validate that each task's training and validation sets contain all necessary classes.

        Parameters
        ----------
        num_tasks : int
            Number of tasks in the model. Should be Y.shape[1].
            
        Y_train, Y_val : torch.Tensor
            Training and validation target tensors of shape (n_samples, num_tasks).

        Returns
        -------
        bool
            True if all tasks have balanced classes in both training and validation sets, False otherwise.
        """
        for task_idx in range(num_tasks):
            train_classes = torch.unique(Y_train[:, task_idx])
            val_classes = torch.unique(Y_val[:, task_idx])
            # Ensure each task has at least two classes (for binary/multi-class classification)
            if len(train_classes) < 2 or len(val_classes) < 2:
                return False
        return True

    def eval_loss(self, X, Y, deterministic=False):
        """
        Compute the total loss, including reconstruction, KL divergence, and task-specific losses.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples, n_tasks)
            True labels for each task.

        deterministic : bool, default=True
            If `True`, uses the latent mean (`mu`) for predictions, ensuring deterministic outputs.
            If `False`, samples from the latent space using the reparameterization trick, introducing stochasticity.

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
            
        auc_scores: list of float
            List of AUC scores for each task.
        """
        X = self._check_tensor(X).to(DEVICE)
        Y = self._check_tensor(Y).to(DEVICE)
        self.eval()
        # Forward pass
        with torch.no_grad():
            recon, mu, logvar, z, task_outputs = self(X, deterministic=deterministic)

            
            total_loss, recon_loss, kl_loss, task_loss, per_task_losses, auc_scores = self.compute_loss(
                X, recon, mu, logvar, z, task_outputs, Y, 
                beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas
            )

        # Convert losses to NumPy arrays
        return (total_loss.item() / len(X),
                recon_loss.item() / len(X),
                kl_loss.item() / len(X),
                task_loss.item(),
                auc_scores,
                )
    
    def eval_recon_loss_with_pred(self, X, recon, mu, logvar):
        """
        Evaluates reconstruction loss and KL divergence loss for given data.
        This method computes the reconstruction loss and KL divergence loss in evaluation mode
        with gradient computation disabled. The losses are normalized by batch size.
        Parameters
        ----------
        X : torch.Tensor
            Input data tensor to reconstruct
            
        recon : torch.Tensor
            Reconstructed data tensor
            
        mu : torch.Tensor
            Mean of the latent distribution
            
        logvar : torch.Tensor
            Log variance of the latent distribution
            
        Returns
        -------
        tuple
            A tuple containing:
            - recon_loss (float): Mean reconstruction loss per sample
            - kl_loss (float): Mean KL divergence loss per sample
        """
        X = self._check_tensor(X).to(DEVICE)
        self.eval()
        # Forward pass
        with torch.no_grad():
            
            recon_loss, kl_loss = self.compute_recon_loss(X, recon, mu, logvar)
            recon_loss = recon_loss.detach().item() / len(X)
            kl_loss = kl_loss.detach().item() / len(X)
            
        # Convert losses to NumPy arrays
        return recon_loss, kl_loss
    
    def eval_recon_loss(self, X):
        """
        Evaluates reconstruction loss and KL divergence loss for given data.
        This method computes the reconstruction loss and KL divergence loss in evaluation mode
        with gradient computation disabled. The losses are normalized by batch size.
        Parameters
        ----------
        X : torch.Tensor
            Input data tensor to reconstruct

        Returns
        -------
        tuple
            A tuple containing:
            - recon_loss (float): Mean reconstruction loss per sample
            - kl_loss (float): Mean KL divergence loss per sample
        """
        X = self._check_tensor(X).to(DEVICE)
        self.eval()
        
        with torch.no_grad():
            recon, mu, logvar, z = self.vae(X) # Forward pass
            recon_loss, kl_loss = self.compute_recon_loss(X, recon, mu, logvar)
            recon_loss = recon_loss.detach().item() / len(X)
            kl_loss = kl_loss.detach().item() / len(X)
            
        return recon_loss, kl_loss
