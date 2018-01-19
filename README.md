# A Simple Neural Attentive Meta-Learner implementation in PyTorch

A PyTorch implementation of the SNAIL building blocks.

This module implements the three blocks in [_A Simple Neural Attentive
Meta-Learner_](https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW) by Mishra et al.

The three building blocks are the following:
- A dense block, built with causal convolutions.
- A TC Block, built with a stack of dense blocks.
- An attention block, similar to the attention mechanism described by Vaswani et al (2017).
