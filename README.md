# (Yet another) PyTorch Tutorial

![Dynamic TOML Badge: Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbpesquet%2Fpytorch-tutorial%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&logoColor=white&logoSize=auto&label=Python&labelColor=%233776AB&color=black)
![Dynamic TOML Badge: PyTorch](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbpesquet%2Fpytorch-tutorial%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.dependencies.torch&logo=pytorch&logoColor=white&logoSize=auto&label=PyTorch&labelColor=%23EE4C2C&color=black)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/bpesquet/pytorch-tutorial/ci.yaml)

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

```bash
git clone https://github.com/bpesquet/pytorch-tutorial.git
cd pytorch-tutorial
poetry install
python {path to Python example file}
```

## Development notes

### Toolchain

This project is built with the following software:

- [Poetry](https://python-poetry.org/) for dependency management;
- [Black](https://github.com/psf/black) for code formatting;
- [Pylint](https://github.com/pylint-dev/pylint) to detect mistakes in the codebase;
- [pytest](https://docs.pytest.org) for testing examples;
- a [GitHub Action](.github/workflows/ci.yaml) for validating the code upon each push;
- [Marp](https://marp.app/) for showcasing `README` files as slideshows during lectures or labs.

### Useful commands

```bash
# Reformat all Python files
black .

# Check the codebase for mistakes
pylint pytorch_tutorial/*

# Run all code examples as unit tests
# The optional -s flag prints code output
pytest [-s]
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2025-present [Baptiste Pesquet](https://bpesquet.fr).
