# (Yet another) PyTorch Tutorial

[![Dynamic TOML Badge: Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbpesquet%2Fpytorch-tutorial%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&logo=python&logoColor=white&logoSize=auto&label=Python&labelColor=black&color=blue)](pyproject.toml)
[![Dynamic TOML Badge: PyTorch](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbpesquet%2Fpytorch-tutorial%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.dependencies.torch&logo=pytorch&logoColor=white&logoSize=auto&label=PyTorch&labelColor=black&color=red)](https://pytorch.org)
[![CI](https://github.com/bpesquet/pytorch-tutorial/workflows/CI/badge.svg)](https://github.com/bpesquet/pytorch-tutorial/blob/main/.github/workflows/ci.yaml)

This repository provides concise and annotated examples for learning the basics of [PyTorch](https://pytorch.org).

> [About this project](ABOUT.md)

## Table of Contents

- [Fundamentals](pytorch_tutorial/fundamentals/)
- [Linear Regression](pytorch_tutorial/linear_regression/)
- [Logistic Regression](pytorch_tutorial/logistic_regression/)
- [MultiLayer Perceptron](pytorch_tutorial/multilayer_perceptron/)
- [Convolutional Neural Network](pytorch_tutorial/convolutional_neural_network/)
- ... (more to come)

## Usage

> [uv](https://docs.astral.sh/uv/) needs to be available on your system.

```bash
git clone https://github.com/bpesquet/pytorch-tutorial.git
cd pytorch-tutorial
uv sync
uv run python {path to example file}
```

## Development notes

### Toolchain

This project is built with the following software:

- [uv](https://docs.astral.sh/uv/) for project management;
- [ruff](https://docs.astral.sh/ruff/) for code formatting and linting;
- [pytest](https://docs.pytest.org) for testing.

### Useful commands

```bash
# Format all Python files
uvx ruff format

# Lint all Python files and fix any fixable errors
uvx ruff check --fix

# Run all code examples as unit tests.
# The optional -s flag prints code output.
uv run pytest [-s]
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2025-present [Baptiste Pesquet](https://bpesquet.fr).
