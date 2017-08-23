# Binary networks from tensorflow to embedded devices

The goal of this project is to provide a way to take models trained in tensorflow and export them to a format that is suitable for embedded devices: C code and a flat binary format for the weights. This project has a special focus on binary networks and quantization because of their relevance for embedded systems.

## bnn module

This module provides helper functions for creating binary models, namely a binary activation function and a useful `layer` function which includes weight binarization, batch normalization, pooling and activation.

The binary weight training is implemented as in [BinaryNet](https://github.com/MatthieuCourbariaux/BinaryNet).

## tf_export module

This module provides an export function which generates C code and a weight file from a tensorflow model. It also implements a few optimizations:

* Detect binary activations and binary weights created with helper functions
* Combine linear operations between layers (ex: bias addition and batch norm)
* Convert linear operations preceeding a binary activation to thresholds
* Layers with binary input and binary weights use fast binary convolutions
* Optional 8-bit quantization for other layers

## Binary weights

Weight binarization reduces the size of weights by a factor of 32 when compared to 32-bit floating point weights. For most models this results in reduced performance but for some models the loss can be minimal.

Technically convolutions with binary weights require less operations but they can be slower on modern CPUs because additional operations are needed to debinarize the weights (although this is mitigated by the reduced number of memory loads).

## Binary convolution

Binary convolutions are possible when both the input and weights are binary. 1-bit multiplications are implementated with XOR operations and accumulated with bitcount operations. This leads to very fast operation. For example, a single SIMD instruction on ARM can perform 128 such 1-bit multiplications.

## Quantization

The quantization implemented in this project is relatively simple, and very much like [this](https://www.tensorflow.org/performance/quantization). It is applied after training unlike the binary weights and activations which require modifications at training time.

Quantization can be especially interesting in the case of binary weights because the precision loss going from 32-bit float input to 8-bit quantized input is very low when combined with the low precision weights. This allows a compromise somewhere between full binarization and binary weights.

## Special convolutions

There are 4 types of "special" convolutions supported by this project:

* **Int8**: The quantized convolution. Uses 8-bit inputs and 8-bit weights and outputs 32-bit integer values.
* **Float with binary weights**: The "BinaryConnect" / "BWN" convolution. Drastically reduces weight size but can be slower than regular convolution on modern CPUs (the weights can be decompressed in that case but then the memory usage remains high).
* **Int8 with binary weights**: The quantized version of the "BinaryConnect" / "BWN" convolution. The weight size is the same but the speed is better.
* **Binary**: The "BinaryNet" / "XNORnet" convolution. Very fast.

### Implementation

The convolution function is optimized for the case of a fully-connected layer with a batch size of 1. "Fast" versions of these operations are implemented using NEON instrinsics for ARM CPUs.

* There is a significant (to the order of 25%) improvement that could be gained using assembly implementations.

### Analysis

2 ops = 1 equivalent multiply-add operation

| Convolution type | Weight bits | Gops on Nexus 5 | Gops on RPi 3 |
| --- | --- | --- | --- |
| Float | 32 | 0.617 | 1.14 |
| Int8 | 8 | 9.99 | 4.06 |
| Float-binary | 1 | 3.18 | 2.39 |
| Int8-binary | 1 | 12.6 | 4.21 |
| Binary | 1 | 62.4 | 38.8 |

* Single thread performance
* Nexus 5: 2.3 GHz Krait 400
* RPi 3: 1.2 GHz Cortex-A53

## Examples

### BinaryNet CIFAR10

This example reimplements the CIFAR10 model from the BinaryNet paper. It also contains a basic example (`test_cifar10.c') to verify the exported weights and code. It has fully binary weights and activations and achieves an accuracy of around 88.6%.

[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

### XNOR-net

This example reimplements the BWN and XNORNet networks from the [XNOR-Net](https://github.com/allenai/XNOR-Net). The BWN network is AlexNet with binarized weights for all but the first and last layers. The XNORNet network is similar but additionally uses binary activations so that binary convolutions can be used for the middle layers.

This example does not do training, instead it uses the pre-trained weights available here:

[BWN](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/alexnet_BWN.t7)

[XNOR](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/alexnet_XNOR.t7)

[cache](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/cache.tar)

* these are torch files so pytorch is required. CUDA is also needed because the files contain CUDA objects, but the pytorch deserializer (read_lua_file.py) can be modified to treat them as non-CUDA objects to run on a machine without CUDA.

[Demo on Android](android-demo/) : Android demo featuring these two networks applied on camera input (and other features)

[Demo on Raspberry Pi 3](rpi-demo/) : Simple linux demo using a webcam

#### Analysis

| Network | Weight Size (MiB) | Run time on Nexus 5 (ms) | Run time on RPi 3 (ms) |
| --- | --- | --- | --- |
| XNORNet | 22.7 | 102 | 216 |
| BWN | 22.8 | 623 | 970 |
| XNORNet (quantized) | 10.9 | 60 | 123 |
| BWN (quantized) | 11.0 | 176 | 546 |

* Measured with `test_xnornet` program
* Single thread run times
* Times for Nexus 5 are best of multiple runs
* For XNORNet the quantization only affects the first and last layers
* TODO pi 64


## References

* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
* [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)
