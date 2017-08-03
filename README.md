# Binary networks from tensorflow to embedded devices

## Tensorflow module

The goal of this project take models trained in tensorflow and export them to a format that is suitable for embedded devices: C code and a flat binary format for the weights.

The module provides helper functions for creating binary models and an export function which generates C code and a weight file from a tensorflow model. The export function implements a few optimizations:

* Detect binary layers created with helper functions
* Combine linear operations between layers (ex: bias and batch norm)
* Convert linear operations preceeding a binary activation to thresholds to reduce operations
* 8-bit quantization for non-binary layers

### Convolution implementations

Supports 32-bit ARM with NEON and 64-bit ARM.

## Examples

### BinaryNet CIFAR10

This example reimplements the CIFAR10 model from the BinaryNet paper. It also contains a basic example using the generated weights verify the accuracy using exported weights and code. It achieves an accuracy of almost 90% on CIFAR-10.

[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

### XNOR-net

These examples reimplement the BWN and XNORnet networks from the XNOR-net paper.

It uses the pre-trained weights available here:

[BWN](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/alexnet_BWN.t7)

[XNOR](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/alexnet_XNOR.t7)

[cache](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/cache.tar)

pytorch and CUDA needed to load the weights. The weight files contain CUDA objects but the deserializer (read_lua_file.py) can be modified to treat them as non-CUDA objects to run on a machine without CUDA.

[Demo on Android](android-demo/)

[Demo on Raspberry Pi 3](rpi-demo/)
