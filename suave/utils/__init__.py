import sys
import numpy as np
import random
import torch

from .seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_interactive_environment():
    """
    Detect if the code is running in an interactive environment (e.g., Jupyter Notebook or IPython).

    Returns
    -------
    bool
        True if running in an interactive environment, False otherwise.
    """
    try:
        # Check if running in IPython or Jupyter Notebook
        if hasattr(sys, 'ps1'):  # Standard interactive interpreter (Python REPL)
            return True
        if 'IPython' in sys.modules:  # IPython or Jupyter environment
            import IPython
            return IPython.get_ipython() is not None
    except ImportError:
        pass  # IPython not installed

    return False  # Not an interactive environment

def generate_hidden_dims(hidden_dim, latent_dim, depth, strategy="constant", order="decreasing"):
    """
    Generator for computing dimensions of hidden layers.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the first hidden layer (encoder input or decoder output).
        
    latent_dim : int
        Dimension of the latent space (encoder output or decoder input).
        
    depth : int
        Number of hidden layers.
        
    strategy : str, optional
        Scaling strategy for hidden layer dimensions:
        - "constant" or "c": All layers have the same width.
        - "linear" or "l": Linearly decrease/increase the width.
        - "geometric" or "g": Geometrically decrease/increase the width.
        Default is "constant".
        
    order : str, optional
        Order of dimensions:
        - "decreasing": Generate dimensions for encoder (hidden_dim -> latent_dim).
        - "increasing": Generate dimensions for decoder (latent_dim -> hidden_dim).
        Default is "decreasing".

    Yields
    ------
    tuple of int
        A tuple representing (input_dim_{i}, output_dim_{i}) for each layer.
    """
    if depth < 0:
        raise ValueError("Depth must be non-negative.")
    
    # Generate dimensions based on strategy
    if strategy in ["constant", 'c']:
        dims = np.full(depth + 2, hidden_dim, dtype=int)
        
    elif strategy in ["linear", 'l']:
        dims = np.linspace(hidden_dim, latent_dim, depth + 2, dtype=int)
        
    elif strategy in ["geometric", 'g']:
        dims = hidden_dim * (latent_dim / hidden_dim) ** np.linspace(0, 1, depth + 2)
        dims = dims.astype(int)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Adjust order for encoder or decoder
    if order == "increasing":
        dims = dims[::-1]
        
    elif order != "decreasing":
        raise ValueError(f"Unknown order: {order}. Must be 'decreasing' or 'increasing'.")

    # Generate layer tuples
    for i in range(len(dims)-2):
        yield dims[i], dims[i + 1]

def set_random_seed(seed: int) -> None:
    """Backward compatible alias for :func:`set_seed`."""

    set_seed(seed)

def to_numpy(x):
    """
    Safely convert input to a NumPy array.

    Parameters
    ----------
    x : torch.Tensor, list, or array-like
        Input to be converted to a NumPy array.

    Returns
    -------
    np.ndarray
        Converted NumPy array.

    Notes
    -----
    - If input is a PyTorch Tensor, it will be moved to CPU before conversion.
    - For unsupported types (e.g., None, sparse tensors), an appropriate error is raised.
    """
    if x is None:
        return None
    
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            raise ValueError("Sparse tensors cannot be converted to NumPy arrays directly.")
        return x.detach().cpu().numpy()  # Detach to avoid issues with gradients
    
    try:
        return np.asarray(x)
    except Exception as e:
        raise TypeError(f"Cannot convert input of type {type(x)} to NumPy array: {e}")
    
def check_tensor(X: torch.Tensor):
    """
    Ensures the input is a tensor and moves it to the correct device.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Input data to check and convert if necessary.

    Returns
    -------
    torch.Tensor
        Tensor moved to the specified device.

    Notes
    -----
    This method automatically handles input conversion from numpy arrays to tensors.
    """
    if isinstance(X, torch.Tensor):
        return X.to(DEVICE)
    else:
        return torch.from_numpy(np.asarray(X)).to(DEVICE)

def make_multitask_classification(n_samples=1000,
                                  n_features=20,
                                  n_tasks=3,
                                  task_classes=None,
                                  task_informative=None,
                                  test_size=0.2,
                                  random_state=None):
    """
    Generate synthetic multi-task classification data.

    For each task, independent classification labels are generated using scikit-learn's `make_classification` 
    function. Features from all tasks are horizontally concatenated to form a single input feature matrix.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples to generate, shared across all tasks. Default is 1000.
        
    n_features : int, optional
        Number of features for each task. Default is 20.
        
    n_tasks : int, optional
        Number of classification tasks. Default is 3.
        
    task_classes : list of int, optional
        A list specifying the number of classes for each task.
        For example, [3, 4, 2] indicates task 1 has 3 classes, task 2 has 4 classes, and task 3 has 2 classes.
        If None, all tasks default to binary classification (2 classes).
        
    task_informative : list of int, optional
        A list specifying the number of informative features for each task.
        If None, half of the features (n_features // 2) are set as informative by default.
        
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.2.
        
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    X_train : pandas.DataFrame
        Training set features, horizontally concatenated from all tasks.
    X_test : pandas.DataFrame
        Test set features, horizontally concatenated from all tasks.
    Y_train : pandas.DataFrame
        Training set multi-task labels, one column per task.
    Y_test : pandas.DataFrame
        Test set multi-task labels, one column per task.
    """
    # Delayed imports to minimize unnecessary overhead
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Set default task_classes and task_informative if None
    if task_classes is None:
        task_classes = [2] * n_tasks  # Default to binary classification
    if task_informative is None:
        task_informative = [n_features // 2] * n_tasks  # Default to half of the features as informative

    # Validate parameter lengths
    assert len(task_classes) == n_tasks, "task_classes length must match n_tasks."
    assert len(task_informative) == n_tasks, "task_informative length must match n_tasks."

    # Initialize lists to store features and labels
    X_list = []
    Y_dict = {}

    for i in range(n_tasks):
        X_task, y_task = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=task_informative[i],
            n_classes=task_classes[i],
            random_state=random_state + i if random_state is not None else None
        )
        X_list.append(X_task)
        Y_dict[f"task_{i+1}"] = y_task

    # Combine task-specific features into a single DataFrame
    X = pd.DataFrame(np.hstack(X_list), 
                     columns=[f"feature_{i+1}" for i in range(n_features * n_tasks)])

    # Combine task-specific labels into a single DataFrame
    Y = pd.DataFrame(Y_dict)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test
