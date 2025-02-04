# PyTorch Tutorial

This repository provides concise and annotated examples for learning [PyTorch](https://pytorch.org).

> Adapted from the great but seemingly unmaintained [PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial) by [Yunjey Choi](https://yunjey.github.io/).

## Table of Contents

- [Fundamentals](tutorials/fundamentals/)
- [Linear Regression](tutorials/linear_regression/)
- ... (more to come)

## Usage

```bash
git clone https://github.com/bpesquet/pytorch-tutorial.git
cd pytorch-tutorial
python {path to Python example file}
```

## Development notes

### Toolchain

This project is built with the following software:

- [Poetry](https://python-poetry.org/) for dependency management;
- [Black](https://github.com/psf/black) for code formatting;
- [Pylint](https://github.com/pylint-dev/pylint) to detect mistakes in the code;
- [pytest](https://docs.pytest.org) for testing the code;
- [Marp](https://marp.app/) for showcasing Markdown files as slideshows during labs.

### Useful commands

```bash
# Reformat all Python files
black .

# Check the code for mistakes
pylint tutorials

# Run all code examples as unit tests
# The -s flag prints code output
pytest [-s] .
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2025-present [Baptiste Pesquet](https://bpesquet.fr).
