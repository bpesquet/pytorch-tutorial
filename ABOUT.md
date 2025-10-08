# About this project

Most PyTorch learning resources are either out-of-date, confusing (why so many [official tutorials](https://pytorch.org/tutorials/)?) or unnecessarily complicated.

In my opinion, a true gem for learning PyTorch from scratch was the [PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial) by [Yunjey Choi](https://yunjey.github.io/). Unfortunately, it has been unmaintained for years.

This project aims to follow a similar path by providing a set of bare-bones, self-sufficient, and beginner-friendly examples. They illustrate the core concepts of PyTorch in a step-by-step fashion, from tensor manipulation to the training of Deep Learning models. They strive to keep up with API changes and follow good practices.

Rather than Jupyter notebooks, these examples are provided as Python files for the following reasons:

- it facilitates a global understanding of their structure;
- it simplifies [formatting and linting](https://docs.astral.sh/ruff/), [testing](https://docs.pytest.org), [versioning](https://github.com/bpesquet/pytorch-tutorial/commits/main/) and [continuous integration](.github/workflows/ci.yaml);
- it promotes concision and general code quality.

All examples are heavily commented and accompanied by `README` files for an easier follow-through.

You are very welcome to contribute to the improvement of this project through [ideas](https://github.com/bpesquet/pytorch-tutorial/issues) or [corrections](https://github.com/bpesquet/pytorch-tutorial/pulls).

Happy PyTorch learning :)
