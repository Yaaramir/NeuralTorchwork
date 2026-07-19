# NeuralTorchwork
An artificial neural network developed with the PyTorch framework.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]()

---

## Table of Contents
- [About the project](#goals)
- [Status Quo](#statusquo)
- [Whats's next?](#todo)

## About the project
NeuralTorchwork is one of three simple neural networks created for classification exercises. Each is coded using a different set of frameworks:

- [NeuralScratchwork](https://github.com/Yaaramir/NeuralScratchwork): This network is created with raw Python and only implements NumPy to organize and utalize data in arrays. This repository dictates speed and content of the other two, since it serves as template for the other two.
- [NeuralTorchwork](https://github.com/Yaaramir/NeuralTorchwork): Based on NeuralScratchwork this project makes use of the [PyTorch framework](https://pytorch.org/) developed by Meta's AI Research lab.
- [NeuralFlowwork](https://github.com/Yaaramir/NeuralFlowwork): Based on NeuralScratchwork this project makes use of the [TensorFlow framework](https://www.tensorflow.org/) developed by Alphabet Inc.'s Google Brain Team.

The first goal is to implement a complete network from scratch in ***NeuralScratchwork*** that can be trained and used for simple classification exercises while implementing the PyTorch and TensorFlow solutions simultaneously.

Once that stage is completed, ***NeuralTorchwork*** will be further developed to be deployed for scientific usage within the [OpenFlexure](https://openflexure.org/) project, while ***NeuralFlowwork*** will transformed in an office and smart home scenario.

Since understanding how neural networks work at its core and learning how to use them successfully is and has been the main goal of this project, development does not necessarily follow the fastes or most efficient way, but often takes a detour to fully capture the edges, boundaries, challenges and oportunities the frameworks and underlying paradigms offer.

Idea and architecture of the NeuralScratchwork are conceived and heavily inspired by [Neural Networks from Scratch](https://nnfs.io/) (Kinsley H., Kukiela D., 2020).

## Status Quo
- A simple model with two linear dense layers, ReLU and Softmax activation functions is implemented. CCE has been chosen for loss calculation and Adam as an optimizer.
- A 2D dataset with three classes of dots spiraling around a centerpoint is implemented.

![Spiral Data](./assets/spiral_data.png)
- The network trains for 10k epochs by forward passing, backward passing, gradient calculation and parameter updating

![Training Results](./assets/training_results.png)
- A test dataset is used to evaluate how the model performs after training

![Test Results](./assets/test_results.png)

## What's next?
Since this netwok is at this point following the NeuralScratchwork, development will aim for a PyTorch implementation of its Status Quo: Regularization will be implemented next to avoid overfitting, different model and layer sizes will be experimented on to find a good fitting model and check possible lacks of balance in the calculation.