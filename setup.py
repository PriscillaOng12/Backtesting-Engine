"""Setup script for the Backtesting Engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="backtesting-engine",
    version="1.0.0",
    author="Quantitative Trading Team",
    author_email="quant@example.com",
    description="A professional-grade backtesting engine for quantitative trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/backtesting-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipywidgets>=8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "backtest=backtesting_engine.cli:main",
        ],
    },
)
