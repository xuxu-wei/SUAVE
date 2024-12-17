#%%
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
import platform
import suave 

print(f"curent SUAVE version: {suave.__version__}")

class InstallWithPytorch(install):
    def run(self):
        # 调用 setuptools 的安装逻辑
        install.run(self)

        # 动态安装 PyTorch
        print("Checking environment for PyTorch installation...")
        try:
            import torch
            print(f"PyTorch already installed: {torch.__version__}")
        except ImportError:
            print("PyTorch not found. Installing PyTorch dynamically...")

            # 检测操作系统
            system = platform.system()
            cuda_version = self._detect_cuda_version()

            if system == "Windows":
                torch_package = self._get_pytorch_package(cuda_version, "win")
            elif system == "Darwin":  # macOS
                torch_package = self._get_pytorch_package(cuda_version, "mac")
            elif system == "Linux":
                torch_package = self._get_pytorch_package(cuda_version, "linux")
            else:
                raise RuntimeError(f"Unsupported operating system: {system}")

            # 安装 PyTorch
            os.system(f"{sys.executable} -m pip install {torch_package}")

    def _detect_cuda_version(self):
        """检测 CUDA 版本，返回 CUDA 标记（如 cu118）或 cpu"""
        try:
            if "TORCH_CUDA_ARCH_LIST" in os.environ or os.system("nvidia-smi") == 0:
                print("GPU detected. Checking for CUDA version...")
                return "cu118"  # 假设 CUDA 11.8，可动态调整
            else:
                print("No GPU detected. Using CPU version.")
                return "cpu"
        except Exception:
            print("Error detecting CUDA. Defaulting to CPU.")
            return "cpu"

    def _get_pytorch_package(self, cuda_version, system):
        """根据系统和 CUDA 版本返回适合的 PyTorch 包"""
        base_url = "https://download.pytorch.org/whl"
        if system == "win":
            return f"torch torchvision torchaudio --index-url {base_url}/{cuda_version}"
        elif system == "mac":
            return "torch torchvision torchaudio"  # macOS 默认 CPU 版
        elif system == "linux":
            return f"torch torchvision torchaudio --index-url {base_url}/{cuda_version}"
        else:
            raise RuntimeError(f"Unsupported system: {system}")
        
def read_requirements():
    """从 requirements.txt 文件中读取依赖"""
    with open("requirements.txt") as f:
        return f.read().splitlines()

def read_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

#%%
setup(
    name="suave-ml",
    version=suave.__version__,

    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url="https://github.com/xuxu-wei/SUAVE",
    author="Xuxu Wei",
    author_email="wxxtcm@163.com",
    license="BSD-3-Clause",
    description="A deep learning model (hybrid VAE) implementation for label information-guided dimensionality reduction and multi-task learning.",

    packages=find_packages(include=["suave", "suave.*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords="VAE supervised-dim-reduction multi-task-learning pytorch sklearn deep learning",
    
    cmdclass={
        "install": InstallWithPytorch,
    },
)
