import sys
import numpy as np
import random
import torch
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

def set_random_seed(seed):
    """
    Set random seed for reproducibility across numpy, random, and PyTorch.
    """
    random.seed(seed)  # Python 的随机数生成器
    np.random.seed(seed)  # NumPy 的随机数生成器
    try:
        import torch
        torch.manual_seed(seed)  # PyTorch 的随机数生成器（CPU）
        torch.cuda.manual_seed(seed)  # PyTorch 的随机数生成器（当前 GPU）
        torch.cuda.manual_seed_all(seed)  # PyTorch 的随机数生成器（所有 GPU）
        torch.backends.cudnn.deterministic = True  # 保证每次卷积的结果一致
        torch.backends.cudnn.benchmark = False  # 禁用自动优化
    except:
        pass

# 将输入转换为 NumPy 数组
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()  # 转换 GPU 张量为 NumPy
    return np.asarray(x)

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