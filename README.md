# Application of Tensor Decompositions in Accelerating Convolutional Neural Networks

**Authors:** Jan Corazza, Mislav Stojanović

## Problem

Convolutional neural networks (CNNs) dominate in image recognition and are also used in video analysis, natural language processing, anomaly detection, pharmacology, gaming, and time series prediction. Convolutional layers primarily process pixelated data, and in this article, we demonstrate how tensor decompositions can be applied to reduce the dimensionality of the original tensor and replace it with a series of smaller mappings. The goal is to speed up the training process and reduce the network size in memory.

## Convolutional Neural Networks

Convolutional neural networks are inspired by the structure of the visual cortex in the brain. They use convolution kernels to map image fragments into new features, often followed by a pooling layer that combines multiple features into one.

## CP Decomposition

Using CP decomposition, we can reduce the number of input channels from S to R, perform depthwise separable convolutions with K_r^x and K_r^y, and finally return the number of channels from R to the original T.

## Tucker Decomposition

With Tucker decomposition, we first reduce the number of input channels from S to R3. Then, we perform a convolution with a tensor that has R3 input channels and R4 output channels. Finally, we use another convolution to return to T output channels.

## Tensor Train

Using the Tensor Train format, we reshape the convolution tensor into a matrix, then factorize the dimensions to obtain a high-order tensor. The tensor train consists of a series of small cubic tensors. This allows for more efficient use of multilinear mappings, resulting in more accurate preservation of results.

## Results

The CP, Tucker, and Tensor Train decompositions allow for significant speedup in the training process of convolutional neural networks, as well as a reduction in memory usage.

|                   | Original    | CP          | Tucker      | TT          |
|-------------------|-------------|-------------|-------------|-------------|
| Number of parameters   | 11689512    | 982985 (8.4091%) | 1118478 (9.5682%) | 864960 (7.3995%) |
| Time           | 499.6475s   | 302.5050s   | 287.0560s   |             |
| Accuracy        | 84.21%      | 69.08%      | 75.07%      |             |

## References

1. Garipov, T., Podoprikhin, D., Novikov, A., & Vetrov, D. (2016). Ultimate tensorization: compressing convolutional and FC layers alike. *arXiv preprint arXiv:1611.03214* [https://arxiv.org/abs/1611.03214](https://arxiv.org/abs/1611.03214)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385* [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
3. Kim, Y.-D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2016). Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications. *arXiv preprint arXiv:1511.06530* [https://arxiv.org/abs/1511.06530](https://arxiv.org/abs/1511.06530)
4. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. In *Proceedings of the Conference on Learning Multiple Layers of Features from Tiny Images*.
5. Lebedev, V., Ganin, Y., Rakhuba, M., Oseledets, I., & Lempitsky, V. (2015). Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition. *arXiv preprint arXiv:1412.6553* [https://arxiv.org/abs/1412.6553](https://arxiv.org/abs/1412.6553)
6. LeCun, Y., & Bengio, Y. (1998). Convolutional Networks for Images, Speech, and Time Series. In *The Handbook of Brain Theory and Neural Networks* (pp. 255-258). MIT Press.
7. Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86, 2278-2324. [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)
8. Nakajima, S., Sugiyama, M., & Babacan, S. (2011). Global Solution of Fully-Observed Variational Bayesian Matrix Factorization is Column-Wise Independent. In *Advances in Neural Information Processing Systems* (Vol. 24). [https://proceedings.neurips.cc/paper_files/paper/2011/file/b73ce398c39f506af761d2277d853a92-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2011/file/b73ce398c39f506af761d2277d853a92-Paper.pdf)
9. Oseledets, I. V. (2011). Tensor-Train Decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317. [https://doi.org/10.1137/090752286](https://doi.org/10.1137/090752286)
