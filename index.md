# Running neural networks on embedded systems

## AlexNet

*AlexNet* is a good example of what a real neural network may look like. It has a raw floating point weight size of 238 MiB, and the size can be much larger if using a *tensorflow* checkpoint. The weights can be compressed, but neural network weights typically have high entropy and do not compress (losslessly) very well. If we wanted to build an embedded or mobile application with it, this would mean 238 MiB of storage and memory used for only the weights and the application would need a few seconds to load the weights from storage into memory. The mobile application would also require a 238 MiB download every time the weights need to be updated. For embedded systems, such a weight size can simply make it impossible to use the neural network and for a mobile application it would be pushing the limits of reasonable user expectations.

As for inference time, inference on a single image in *AlexNet* requires over 2 billion floating point operations (1 multiply-add operation counted as 2 operations). In this case a typical smartphone CPU can achieve this at a real-time rate. In fact, my implementation takes around 1 second on a single Nexus 5 core, a 2013 phone. Reducing the amount of work required to perform inference is nevertheless interesting: reducing battery usage, allowing applications that require a more strict definition of real-time and allowing larger models.

## Binarized AlexNet

One method of reducing the weight size and inference time is binarization. Weights can be binarized to contain only sign information, reducing the weight size by a factor of 32. While it may seem like binarizing weights would reduce accuracy drastically, a basic training example using LeNet-5 on MNIST shows that binarizing weights can actually improve accuracy due to better generalization. In addition to binarizing weights, it is possible to use binary activation functions which have a binary output of -1 or +1. This doesn't reduce weight size and reduces accuracy, but it allows for layers with both binary inputs and binary weights, in which case it is possible to use XOR and bitcount operations for the convolution.

[XNOR-Net](https://github.com/allenai/XNOR-Net) provides two pre-trained binarized variations of *AlexNet*, in the form of *Torch* models, which have a size of 476 MiB each and store weights as floating point values. To use these models in *tensorflow*, I first reimplemented both models in *tensorflow* and imported the weights using *pytorch*. The *tensorflow* models however still use floating point values to store the weights and do not have any support for fast binary convolutions. To run the model using the binarized weights I first created my own implementation of the operations required to run these two models in C. Then, I wrote a script which parses the *tensorflow* graph and generates C code which implements the forward pass calling the C functions I implemented.

The first variation, *BWN*, has binarized weights for all intermediary layers. This introduces layers which have floating point inputs and binary weights. The second variation, *XNOR-Net*, also features binary activations. This introduces layers which have both binary inputs and binary weights.

## Adding quantization

Quantization is another method for reducing weight size and improving inference time, and it is already [possible with tensorflow](https://www.tensorflow.org/performance/quantization). However, for this project I am mostly interested in the case where it is combined with binarization. In both **XNOR-Net** models, the first and last layers aren't binarized, which makes them available for quantization. The first layer accounts for a large portion of the total inference time (around half for the *XNOR-Net* model), while the last layer accounts for 16 MiB of the total 23 MiB weight size. By applying quantization to the first and last layers, the weight size is reduced to 11 MiB and the inference time is lowered. Quantization introduces layers with 8-bit input and 8-bit weights.

For the *BWN* model, quantization is also possible for the intermediary layers, where the weights are already binarized but the input values are floating point values. This allows for 8-bit input with 1-bit weight convolutions, which run faster with a negligible difference in the network's output. This strikes a balance of speed and accuracy, somewhere in between float-binary operations and binary-binary operations.

## Android app

To showcase my work, I created an Android App which runs inference on camera input using the selected binarized variation of *AlexNet*. The application also allows capturing clips and playing them back, running inference again on the recorded clip.

![alt](http://i.imgur.com/KrW94y0.jpg)

