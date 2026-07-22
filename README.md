# NeuralTorchwork
An artificial neural network developed with the PyTorch framework.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]()

## Table of Contents
- [About the project](#about-the-project)
- [Status Quo](#status-quo)
- [Whats's next?](#whats-next)

## About the project
NeuralTorchwork is one of three simple neural networks created for classification exercises. Each is coded using a different set of frameworks:

- [NeuralScratchwork](https://github.com/Yaaramir/NeuralScratchwork): This network is created with raw Python and only implements NumPy to organize and utilize data in arrays. This repository dictates the speed and content of the other two, as it serves as the template for both.
- [NeuralTorchwork](https://github.com/Yaaramir/NeuralTorchwork): Based on NeuralScratchwork this project makes use of the [PyTorch framework](https://pytorch.org/) developed by Meta's AI Research lab.
- [NeuralFlowwork](https://github.com/Yaaramir/NeuralFlowwork): Based on NeuralScratchwork this project makes use of the [TensorFlow framework](https://www.tensorflow.org/) developed by Alphabet Inc.'s Google Brain Team.

The first goal is to implement a complete network from scratch in ***NeuralScratchwork*** that can be trained and used for simple classification exercises while implementing the PyTorch and TensorFlow solutions simultaneously.

Once that stage is completed, ***NeuralTorchwork*** will be further developed to be deployed for scientific usage within the [OpenFlexure](https://openflexure.org/) project, while ***NeuralFlowwork*** will transformed in an office and smart home scenario.

Since understanding how neural networks work at its core and learning how to use them successfully is and has been the main goal of this project, development does not necessarily follow the fastes or most efficient way, but often takes a detour to fully capture the edges, boundaries, challenges and oportunities the frameworks and underlying paradigms offer.

Idea and architecture of the NeuralScratchwork are conceived and heavily inspired by [Neural Networks from Scratch](https://nnfs.io/) (Kinsley H., Kukiela D., 2020).

## Status Quo
- ### Model:
  - 2 hidden layers with ReLU
  - 2 dropout layers with 0.1 dropout rate
  - 1 output layer
  - CCE with implemented Softmax

  ![network diagram](./assets/nn_concept.png)

- ### Data:
A training dataset of spiraling points in a 2D space with 1,000 samples and 3 classes is created, using the nnfs.io dataset library.

![Spiral Data](./assets/spiral_data.png)

- ### Training:
The network trains for 10k epochs by performing forward passes, backward passes, gradient calculation, and parameter updating.

![Training progression results](./assets/training_progression_results.png)

- ### Validation:
A validation dataset is used to evaluate model performance after training.

![Training vs Validation Results](./assets/training_vs_validation_results.png)

- ### Evaluation:
  - Results fall a little bit behind those before droput implementation in regards to their absolute scores, while differences between validation an training have narrowed.
  - This indicates strong generalization and avoidance of overfitting and co-adaptation, clearly an effect of L2 and dropout regularization.
  - A small degree of overfitting can still be assumed, since validation accuracy should be higher than training accuracy that is reduced naturally by dropout.

## What's next?
Since this network currently follows the progression of NeuralScratchwork, development will aim for a PyTorch-native implementation of its Status Quo. 
- The Evaluation expresses the need for more hyperparameter tuning, especially after dropout implementation
- Data preparation and preprocessing
- Different model architectures and layer sizes will be experimented with to identify the best-fitting model and detect potential imbalances in the calculations.
Have a look at this project's [Issues](https://github.com/Yaaramir/NeuralTorchwork/issues) as well to see imminent enhancements and bug fixes.


___

[_Jump back to the top_](#neuraltorchwork)