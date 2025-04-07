# Linear Tangent Closed-Form Solution for MLP

This repository implements a linear tangent closed-form solution for fine-tuning a Multi-Layer Perceptron (MLP) on the MNIST dataset.

## Features
- **Model**: An MLP with LeakyReLU activations is used as the base architecture.
- **Closed-Form Solution**: Computes parameter updates using a linear tangent approximation, leveraging the Jacobian computed via `jacrev` from PyTorch's `torch.func` module.
- **Pretrained Models**: Pretrained weights are provided for convenience.
- **Slice-Based Updates**: The parameter updates are applied to slices of the weight matrix, with performance varying based on slice size.

## Performance
The performance of the closed-form solution depends on the `slice_size` parameter, which determines the size of the parameter slices updated in each iteration. Below are the test accuracies observed on MNIST:

- `slice_size = 16`: **90.39%**
- `slice_size = 32`: **93.05%**
- `slice_size = 64`: **9%**