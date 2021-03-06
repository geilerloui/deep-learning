# Deep Learning Nanodegree Foundation

This repository contains material related to Udacity's [Deep Learning Nanodegree Foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) program. It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight intialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by Udacity experts, but they are available here as well.

## Table Of Contents

### 1. Introductions - NumPy

* [Linear Regression](https://github.com/geilerloui/deep-learning/blob/master/linear-regression/Regression.ipynb):
* [Gradient Descent](https://github.com/geilerloui/deep-learning/blob/master/gradient-descent/GradientDescent.ipynb): 
* [Your First Neural Network](https://github.com/udacity/deep-learning/tree/master/first-neural-network): Implement a neural network in Numpy to predict bike rentals.
* [Sentiment Analysis with Numpy](https://github.com/udacity/deep-learning/tree/master/sentiment-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.

### 2. Feedforward Neural Networks - Tensorflow, Keras, TFLearn

* [Lesson 1: Miniflow](): is about Differentiable Graphs, the abstraction that tensorflow use to run and train networks. Miniflow will be our very own version of tensorflow.
* [Lesson 2: Intro to Keras](https://github.com/geilerloui/deep-learning/tree/master/student-admissions-keras): with a mini-project on student admission test
* [Lesson 3: Intro to TFLearn](https://github.com/udacity/deep-learning/tree/master/intro-to-tflearn): A couple introductions to a high-level library for building neural networks.

* Lesson 4: Lessons on Tensorflow: 
  * [Basic Data Types and Functions](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Intro_to_TensorFlow.ipynb): What is ```tf```, ```Session, constant, placeholder, feed_dict, add, subtract ..., cast, Variable, zeros, truncated_normal, Cross Entropy and Softmax```
  * [Mini-batch and Epochs](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Mini-batch_Epochs.ipynb): Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. An Epoch is a single forward and backward pass on the whole dataset. We downloaded the data with ```Keras```
  * [Multilayer Perceptron](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Deep_Neural_Network-1.ipynb) The data will be downloaded with the new feature ```tf.data```
  * [Save and Restore TensorFlow models](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Deep_Neural_Network-2_Save_Restore_models.ipynb)
  * [Loading the Weights and Biases into a new model](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Deep_Neural_Network-3_Loading_weights_in_new_model.ipynb)
  * [Early Termination, Regularization and Dropout](https://github.com/geilerloui/deep-learning/blob/master/Lesson_NeuralNets/Deep_Neural_Network-4_Early_termination_regularization_dropout.ipynb)
  * [Intro to TensorFlow](https://github.com/udacity/deep-learning/tree/master/intro-to-tensorflow): Starting building neural networks with Tensorflow.


### 3. Convolutional Neural Networks

* [Lesson 2: CNN in TensorFlow]()
* [Lesson 3: Convolutional Neural Networks]()
* [Lesson 4: Autoencoders](https://github.com/udacity/deep-learning/tree/master/autoencoder): Build models for image compression and denoising, using feed-forward and convolution networks in TensorFlow.
* [Lesson 5: Transfer Learning (ConvNet)](https://github.com/udacity/deep-learning/tree/master/transfer-learning). In practice, most people don't train their own large networkd on huge datasets, but use pretrained networks such as VGGnet. Here you'll use VGGnet to classify images of flowers without training a network on the images themselves.
* [Lesson 6: Deep Learning for Cancer Detection with Sebastian Thrun]()
* [Lesson 7: Weight Intialization](https://github.com/udacity/deep-learning/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Lesson 8: Batch normalization](https://github.com/udacity/deep-learning/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Project: Image classification](https://github.com/udacity/deep-learning/tree/master/image-classification): Build a convolutional neural network with TensorFlow to classify CIFAR-10 images.

### 4. Recurrent Neural Networks

* [Lesson 1: Recurrent Neural Networks](): XX
* [Lesson 2: Long Short-Term Memory Networks (LSTM)](): XX
* [Lesson 3: Implementation of RNN and LSTM](https://github.com/udacity/deep-learning/tree/master/intro-to-rnns): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text.
* [Lesson 4: Hyperparameters](): In this lesson, we'll look at number of different hyperparameter that are important for our deep learning to work. We'll discuss starting values and intuitions for tuning each hyperparameter.
* [Lesson 5: Embeddings (Word2Vec)](https://github.com/udacity/deep-learning/tree/master/embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Lesson 6: Sentiment Analysis RNN](https://github.com/udacity/deep-learning/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if a text sample is positive or negative.
* [Project 1: Text Generation](https://github.com/udacity/deep-learning/tree/master/tv-script-generation): Train a recurrent neural network on scripts from The Simpson's (copyright Fox) to generate new scripts.
* [Lesson 7: Sequence to sequence](https://github.com/udacity/deep-learning/tree/master/seq2seq): Implement a sequence-to-sequence recurrent network.
* [Project 2: Machine Translation](https://github.com/udacity/deep-learning/tree/master/language-translation): Train a sequence to sequence network for English to French translation (on a simple dataset)

### 5. Generative Adversarial Networks

* [Lesson 1: Generative Adversatial Network on MNIST](https://github.com/udacity/deep-learning/tree/master/gan_mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Lesson 2: Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [Lesson 3: Semi-Supervised GAN](https://github.com/geilerloui/deep-learning/tree/master/semi-supervised)
* [Project: Face Generation](https://github.com/udacity/deep-learning/tree/master/face_generation): Use a DCGAN on the CelebA dataset to generate images of novel and realistic human faces.

### 6. Reinforcement Learning

* [Lesson 1: Introduction to RL](https://github.com/geilerloui/deep-learning/blob/master/reinforcement-learning/Lesson-1.md): Reinforcement learning is a type of Machine Learning where the machine or software agent learns how to maximize its performance at a task.
* [Lesson 2: The RL Framework: The Problem](https://github.com/geilerloui/deep-learning/blob/master/reinforcement-learning/Lesson-2.md): Learn how to mathematically formulate tasks as Markov Decision Processes
* [Lesson 3: The RL Framework: The Solution](https://github.com/geilerloui/deep-learning/blob/master/reinforcement-learning/Lesson-3.md): XX
* [Lesson 4: Dynamic Programming](): The dynamic programming setting is a useful first step towards tackling the reinforement learning problem.
* [Exercice Lesson 4: Dynamic Programming](https://github.com/geilerloui/deep-learning/blob/master/reinforcement-learning/Exercice_Lesson-4/Dynamic_Programming.ipynb): Execise with OpenAI Gym: FrozenLakeEnv
* [Lesson 5: Monte Carlo Methods](): Write your own implementation of Monte Carlo control to teach an agent to play Blackjack ! With OpenAI Gym: BlackjackEnv
* [Lesson 6: Temporal-Difference Methods](): Learn about how to apply temporal-difference methods such as Sarsa, Q-Learning, and Expected Sarsa to solve both episodic and continuous tasks.
* [Lesson 7: Solve OpenAI Gym's Taxi-v2 Task](): With reinforcement learning now in your toolbox, you're ready to explore a mini project using OpenAI Gym!

### 7. Deep Reinforcement Learning

* [Lesson 8: RL in Contunous Spaces](): Review the fundamental concepts of reinforcement learning, and learn how to adapt traditional algorithms to work with continuious spaces.
* [Lesson 9: Deep Q-Learning](): Extend value-based reinforcement learning methods to complex problems using deep neural networks.
* [Lesson 10: Policy-Based Methods](): Policy-based methods try to directly optimize for the optimal policy. LEarn, how they work, and why they are important, especially for domains with continuous action spaces.
* [Lesson 11: Actor-Critic Methods](): Learn how to combine value-based and policy-based methods, bringing together the best of both world, to solve challenging reinforcement learning problems.
* [Lesson 12: Teach a Quadcoptere How to Fly](): Build a quadcopter flying ageent that learns to take off, hover and land using reinforcement learning
* [Project: Reinforcement Learning (Q-Learning)](https://github.com/udacity/deep-learning/tree/master/reinforcement): Implement a deep Q-learning network to play a simple game from OpenAI Gym.

### Tools

* [Tensorboard](https://github.com/udacity/deep-learning/tree/master/tensorboard): Use TensorBoard to visualize the network graph, as well as how parameters change through training.


## Datasets
* [notMNIS](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html): consists of images of a letter from A to J in different fonts.
* [MNIST](https://fr.wikipedia.org/wiki/Base_de_donn%C3%A9es_MNIST): is a set of images from 0 to 9


## Bibliography
* [The Deep Learning Textbook](http://www.deeplearningbook.org/), from Ian GoodFellow ...
* [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
* [STAT 946](https://www.youtube.com/watch?v=XTWPyW2mTUg&list=PLehuLRPyt1HxTolYUWeyyIoxDabDmaOSB) by Ali Ghodsi Waterloo University
* [CS231n Lecture 4 Backpropagation](https://www.youtube.com/watch?v=59Hbtz7XgjM) Computational Graph
* [Andrej Karpathy on Backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
* [History of Deep Learning](https://www.youtube.com/watch?v=ht6fLrar91U), by Frank Chen
* [Reinforcement learning: An Introduction, by Andrew Barto and Richard Sutton](https://www.youtube.com/watch?v=ht6fLrar91U), by Frank Chen


## Dependencies

Each directory has a `requirements.txt` describing the minimal dependencies required to run the notebooks in that directory.

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.

### Conda Environments

You can find Conda environment files for the Deep Learning program in the `environments` folder. Note that environment files are platform dependent. Versions with `tensorflow-gpu` are labeled in the filename with "GPU".
