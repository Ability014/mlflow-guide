"""Package setup for MLflow Iris Enterprise."""

from setuptools import setup, find_packages

setup(
    name="mlflow-iris-enterprise",
    version="1.0.0",
    description="Enterprise MLflow reference implementation",
    author="ML Platform Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "mlflow>=2.9.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=src.models.train_model:main",
        ],
    },
)