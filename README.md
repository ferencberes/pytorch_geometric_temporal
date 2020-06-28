[pypi-image]: https://badge.fury.io/py/torch-geometric-temporal.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric-temporal
[build-image]: https://travis-ci.com/benedekrozemberczki/pytorch_geometric_temporal.svg?branch=master
[build-url]: https://travis-ci.com/benedekrozemberczki/pytorch_geometric_temporal
[docs-image]: https://readthedocs.org/projects/pytorch-geometric-temporal/badge/?version=latest
[docs-url]: https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/benedekrozemberczki/pytorch_geometric_temporal/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/benedekrozemberczki/pytorch_geometric_temporal?branch=master

<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](https://pytorch-geometric-temporal.readthedocs.io)** | **[External Resources](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/resources.html)**

**Methods Included**

In detail, the following temporal graph neural networks were implemented.

**Graph Convolutional Recurrent Networks**

**Discrete Stacked Dynamic Graph Neural Networks**

**Discrete Integrated Dynamic Graph Neural Networks**

**Continuous Recurrent Neural Network Based Graph Neural Networks**

**Continuous Point Process Based Graph Neural Networks**


## Installation

To install the binaries for PyTorch 1.5.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.


## Running tests

```
$ python setup.py test
```

