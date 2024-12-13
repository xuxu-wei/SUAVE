from setuptools import setup, find_packages

setup(
    name="suave",
    version="0.1.0a1",

    long_description=open("README.md").read(),
    long_description_content_type="Hybrid Variational Autoencoder with Multi-Task Learning (Alpha Stage)",
    url="https://github.com/xuxu-wei/SUAVE",
    author="Xuxu Wei",
    author_email="wxxtcm@163.com",
    license="BSD-3-Clause",

    packages=find_packages(include=["suave", "suave.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.18.0",
        "scikit-learn>=1.0.2",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "tqdm>=3.0.0",
    ],

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

    keywords="VAE supervised-dim-reduction multi-task-learning pytorch sklearn deep machine learning",
)
