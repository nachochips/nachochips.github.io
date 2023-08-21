---
layout: post
title: Pix2Pix Class Week 1
tags:
  - studylog
  - Python
  - Pytorch
  - Pix2Pix
  - Deeplearning
use_math: true
comments: true
---

Exploring Machine Learning Basics 🚀

Before moving forward, I felt the need to take a step back and lay some foundation. 
So, I decided to rewind a bit and revisit the materials from the previous classes to grasp the fundamental concepts.

## Grasping the Fundamentals

In this section, I will pile up some important basic concepts that I didn't knew and felt the necessities to fill the gaps. 
As a beginner, I'm all ears for your feedback and suggestions. If you notice any inaccuracies or feel that certain concepts require more clarification, feel free to share your thoughts in the comments below. 💬  

### 1. Loss Function 

In machine learning, a loss function (also known as a cost function or objective function) is a mathematical function that measures how well a machine learning model's predictions match the actual target values in the training data. The goal of a machine learning model is to minimize this loss function. In simpler terms, the loss function quantifies how "wrong" the model's predictions are.

<img src="/docs/assets/linearlog.png" width="80%" height="80%">
[Source: https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/]

In regression tasks (predicting continuous values), a common loss function is the Mean Squared Error (MSE). It calculates the average squared difference between predicted and actual values.
In classification tasks (categorizing into classes), the Cross-Entropy Loss (also called Log Loss) is often used. It measures the dissimilarity between predicted probabilities and the true class labels.

The process of training a machine learning model involves finding the parameters (weights and biases) that minimize the chosen loss function. Optimization algorithms, such as gradient descent, are used to adjust these parameters iteratively to reduce the loss.

Remember, the ultimate goal is to find the model's parameters that lead to the smallest loss value, which means the model's predictions are as close as possible to the actual target values.

#### Differentiation of the Loss Function 

Think of the loss function as a guide that tells the machine learning model how well it's doing. The goal of the model is to follow this guide and get better over time.

Imagine you're hiking up a mountain, and your goal is to reach the lowest point (the bottom of a valley). The loss function is like a map that shows you how high you are on the mountain at each point. Your job is to find a path that takes you downhill, closer to the lowest point.

Now, the important thing is that you need to be able to tell which direction is downhill at any given point. This is where differentiability comes in. If the map (loss function) is smooth and continuous, it means you can tell the slope or steepness at any point. This helps you know which way to go to get closer to the bottom.

If the map suddenly becomes very bumpy or has cliffs where there are gaps, it's hard to figure out which direction to go. In the same way, if the loss function isn't smooth and differentiable, the machine learning model can't tell how to improve because it doesn't know which way to adjust its predictions.

So, having a loss function that can be differentiated is like having a clear map that guides the model in the right direction to improve its predictions. It's a bit like the model's sense of direction, helping it find its way to better results.

**Related terms: Optimization Algorithms, Gradient Descent, Efficient Convergence, Backpropagation**

### 2. Backpropagation, Forwardpropagation

Forward Propagation:
Imagine you're baking a cake. You start with some ingredients (inputs) like flour, sugar, and eggs. You follow a recipe (model) that tells you how to mix and bake these ingredients to get the final cake (output). In machine learning terms, this process is called forward propagation.

When you use your model to make predictions, you're actually doing forward propagation. The inputs go through the layers of your model, which perform calculations using weights and biases. These calculations transform the inputs step by step, just like mixing the cake ingredients. The final result is the model's prediction.

Backpropagation:
Now, let's say someone tastes your cake and tells you if it's good or needs improvement. Backpropagation is like learning from this feedback to become a better baker.

When your model makes predictions, it's not always perfect. Backpropagation is the process of figuring out how wrong your predictions are and adjusting the model's "recipe" (weights and biases) to make better predictions next time.

Here's how it works: Imagine you're baking many cakes, and someone tells you how far off your cake's taste is from the perfect cake. Backpropagation helps you figure out which ingredient (or which part of the model) contributed the most to the difference. Then you adjust your recipe slightly to make the cake taste better next time.

In the same way, backpropagation calculates how much each part of your model contributed to the prediction error. It then adjusts the weights and biases in the model's layers to reduce that error. This way, the model learns from its mistakes and becomes better at making predictions over time.

So, in short:

Forward propagation: Like following a recipe to bake a cake (making predictions).
Backpropagation: Learning from feedback to adjust the recipe and bake a better cake next time (improving the model).

### 3. ReLU

ReLU - Rectified Linear Activation:
Think of ReLU as a light switch for a neuron in a neural network. It determines whether the neuron should "fire" or not based on the input it receives. Here's how it works:

Input Evaluation: When an input value is passed to a neuron in a neural network, ReLU checks the value:
If the input is positive or zero, ReLU lets the value pass unchanged.
If the input is negative, ReLU "turns off" the neuron by setting the output to zero.
In mathematical terms, the ReLU function can be defined as: f(x) = max(0, x)

<img src="/docs/assets/reLU.png" width="80%" height="80%">
[Source: https://yeowool0217.tistory.com/502]

Benefits of ReLU:
ReLU is popular in deep learning for a few key reasons:

1) Simplicity: ReLU is a simple function that's computationally efficient to compute.

2) Non-linearity: While it looks linear (a straight line), it introduces non-linearity to the model. This allows neural networks to learn and represent more complex patterns in data.

3) Avoiding Vanishing Gradient Problem: ReLU helps mitigate the vanishing gradient problem that can occur with other activation functions like sigmoid or tanh. This problem can make training deep networks more challenging.

  The vanishing gradient problem occurs primarily with activation functions that have small derivatives in certain     input ranges. For example, sigmoid and tanh activation functions squash their inputs into a limited range (between   0 and 1 for sigmoid, and between -1 and 1 for tanh). As the inputs move toward the extreme ends of these ranges,     the derivatives become close to zero. This means that when gradients are backpropagated through many layers, they    diminish significantly. Consequently, the early layers of the network receive extremely small updates, which can     slow down or even stall the learning process for those layers.

  ReLU, on the other hand, doesn't suffer from the vanishing gradient problem to the same extent as sigmoid and tanh   activations. When the input to a ReLU neuron is positive, the derivative is 1 (or simply non-zero), which means      the gradient doesn't vanish. This encourages larger gradients to flow backward during training.

  However, it's important to note that while ReLU addresses the vanishing gradient problem, it can introduce a         different issue known as the "dying ReLU" problem. This occurs when a ReLU neuron always outputs zero due to its     weights being adjusted in a way that only negative values are passed through it. Variations of ReLU, like Leaky      ReLU and Parametric ReLU, have been developed to mitigate this problem while still benefiting from the non-          vanishing gradient property.

   In Leaky ReLU, when the input is negative, the function allows a small, non-zero output. This non-zero output for negative inputs helps prevent the "dying ReLU" problem.

In mathematical terms, the Leaky ReLU function can be defined as:

For $x >= 0: f(x) = x$

For $x < 0: f(x) = α * x$ (where $α$ is a small positive constant, usually around 0.01)

<img src="/docs/assets/leakyreLU.png" width="80%" height="80%">
[Source: (https://medium.com/aaweg-i-nterview/why-leaky-relu-is-less-sensitive-to-initialization-than-relu-activation-function-86c0a79ba425)]

4) Sparsity: ReLU's "turn off" behavior (outputting zero for negative inputs) creates sparsity in the network, meaning fewer neurons are activated. This can lead to more efficient and faster training.

Visual Representation:
Imagine a neuron as a switch that turns on when the input is positive or zero and turns off when the input is negative. Visually, this looks like a ramp that starts at zero and goes upwards for positive values.

In simple terms, ReLU helps neural networks make decisions by considering whether the inputs are positive or negative. If they're positive or neutral (zero), the neuron fires, and if they're negative, the neuron stays quiet.

### 4. Convolution
In the context of image processing, convolution is used to extract features from an image by applying a filter or a kernel to the image. The kernel is a small matrix of values that is moved over the input image in a sliding manner, and at each position, it performs an element-wise multiplication with the corresponding pixels in the image. The results of these multiplications are summed up, and the sum is placed in the output image at the corresponding position. This process effectively highlights certain patterns or features in the image.

Convolutional neural networks (CNNs) leverage this concept of convolution to automatically learn features from images. The network learns to adjust the values of the kernels during training to recognize different patterns in the input images, making CNNs highly effective for tasks like image classification, object detection, and more.

Two videos on Youtube really helped me to understand this concept that was really difficult to embrace as a beginner in this filed (and mathematics): 

1) Convolutional Neural Networds (CNNs) explained (deeplizard): https://www.youtube.com/watch?v=YRhxdVk_sIs
2) But what is a convolution? (3Blue1Bronw): https://youtube.com/watch?v=KuXjwB4LzSA

**#Related terms: padding, strides, and multiple channels**

