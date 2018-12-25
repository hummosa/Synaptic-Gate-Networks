# Synaptic-Gate-Networks

Synaptic Gate Networks
Feedback connection are abundant in biological networks, but their exact role and how they interact with feedforward propagation remains unclear.

We here propose an iterative feed forward model that has an additional  output in the form of an embedding z, that is used to modulate  the weights of earlier layers. This allows later layers in the network to  re-define the function computed by the network on the next feed forward pass. This mechanism of top-down control provides substrates for Bayesian inference, attentional control, and multi-modal perception.

The model is implemented in tensorflow Eager execution mode using a custom Keras layer. This allows for dynamic control of feedforward and feedbackward propagation in a recursive  network.

A simplified unrolled version was also implement as a Tensorflow  graph model for rapid multi-GPU training.
