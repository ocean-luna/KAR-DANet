# ResNet-Kolmogorov-Arnold Network

This repository contains an efficient implementation of the ResNet-Kolmogorov-Arnold Network (ResNet-KAN). The original implementation of ResNet-KAN can be found [here](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is primarily due to the need to expand all intermediate variables to perform various activation functions. For a layer with `in_features` inputs and `out_features` outputs, the original implementation requires expanding the input to a tensor with the shape `(batch_size, out_features, in_features)` to perform the activation functions. However, all activation functions are linear combinations of a fixed set of basis functions, which are B-splines; given this, we can reformulate the computation to activate the input with different basis functions and then linearly combine them. This reformulation can significantly reduce memory costs and simplify the computation to a straightforward matrix multiplication, which naturally works for both forward and backward passes.

The issue lies in **sparsification**, which is claimed to be crucial for the interpretability of KAN. The authors proposed an L1 regularization defined on the input samples, which requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and thus is not compatible with the reformulation. I replaced the L1 regularization with an L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation. The author's implementation indeed includes this type of regularization alongside the one described in the paper, so I believe it might help. More experiments are needed to verify this; but at least the original approach is infeasible if efficiency is desired.

Another difference is that, in addition to the learnable activation functions (B-splines), the original implementation also includes a learnable scale for each activation function.

**Note**: The Hyper Kvasir dataset is a public dataset used for testing and validating the performance of machine learning models, especially in the fields of signal processing and pattern recognition.

