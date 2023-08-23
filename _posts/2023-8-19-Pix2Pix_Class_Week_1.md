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

Imagine neural networks as a complex network of interconnected neurons, akin to the human brain. Now, think of ReLU (Rectified Linear Activation) as a key player in this intricate web, acting like a switch that decides whether a neuron should light up or remain dormant based on the incoming signals. Let's delve into this concept, shedding light on the mechanics of ReLU.

#### The Decision Process:
When a value enters a neuron within the neural network, ReLU steps in to evaluate it. Here's how the process unfolds:

1) Positive or Zero?: If the input value is positive or zero, ReLU lets it pass through untouched. It's like giving the green light to information flow.

2) Negative?: However, if the input value is negative, ReLU flips the switch and sets the neuron's output to zero. Essentially, it silences the neuron, ensuring it doesn't contribute to the network's decision-making.

Mathematical Magic:
This process can be translated into mathematical terms using a simple formula: $f(x) = max(0, x)$. In essence, it captures the idea of letting positive values flow freely while blocking negative ones.

<img src="/docs/assets/reLU.png" width="80%" height="80%">
[Source: https://yeowool0217.tistory.com/502]

#### Benefits of ReLU:
ReLU is popular in deep learning for a few key reasons:

1) Simplicity: ReLU is a simple function that's computationally efficient to compute.

2) Non-linearity: While it looks linear (a straight line), it introduces non-linearity to the model. This allows neural networks to learn and represent more complex patterns in data.

3) Avoiding Vanishing Gradient Problem: ReLU helps mitigate the vanishing gradient problem that can occur with other activation functions like sigmoid or tanh. This problem can make training deep networks more challenging.

4) Sparsity: ReLU's "turn off" behavior (outputting zero for negative inputs) creates sparsity in the network, meaning fewer neurons are activated. This can lead to more efficient and faster training.

#### Leaky ReLU: A Repair Mechanism:
Leaky ReLU offers a remedy to the "dying ReLU(vanishing gradient problem)" concern. If inputs are negative, Leaky ReLU permits a small, non-zero output. This keeps the neuron alive even in the face of negative values.

Mathematically, Leaky ReLU can be defined as:

For $x \geq 0$: f(x) = x
For $x < 0$: f(x) = α * x (where $α$ is a small positive constant, typically around 0.01)

<img src="/docs/assets/leakyreLU.png" width="80%" height="80%">
[Source: (https://medium.com/aaweg-i-nterview/why-leaky-relu-is-less-sensitive-to-initialization-than-relu-activation-function-86c0a79ba425)]

In simple terms, ReLU helps neural networks make decisions by considering whether the inputs are positive or negative. If they're positive or neutral (zero), the neuron fires, and if they're negative, the neuron stays quiet.

### 4. Convolution
Imagine an image as a puzzle of pixels, each representing a tiny fragment of the picture. Convolution involves sliding a small grid, called a kernel or filter, over this image. This kernel is like a stencil with numbers on it. As the kernel travels across the image, it multiplies its numbers with the corresponding pixel values beneath it. The magic happens when these multiplied values are added up. This summation is then placed into a new image, creating what's known as a feature map. This process might seem perplexing, but it's actually a way to highlight particular patterns or details within an image.

Now, here's where it gets even more intriguing. Convolution isn't just about image manipulation; it's also a fundamental concept behind Convolutional Neural Networks, or CNNs. These networks are like virtual artists, learning to identify distinct shapes, textures, and objects in images. During their training, CNNs adjust the numbers in those kernels, enabling them to recognize various patterns in the images they encounter. This remarkable ability makes CNNs exceptionally powerful for tasks like telling apart different objects in pictures, spotting specific items within larger scenes, and much more.

If you're looking to delve deeper, two YouTube videos were game-changers for me when I was grasping this intricate concept from scratch:

1) Convolutional Neural Networks (CNNs) Explained (Deeplizard): This video offers a clear breakdown of CNNs and how they utilize convolution for image analysis. Watch it here: Convolutional Neural Networks (CNNs) explained (Deeplizard).

2) But What Is Convolution? (3Blue1Brown): A fantastic video that delves into the heart of convolution, exploring the very essence of this concept. Check it out here: But what is a convolution? (3Blue1Brown).

**#Related terms: padding, strides, and multiple channels**

### 5. Skip Connection
Imagine you're trying to build a tower of building blocks to reach a certain height. Each block represents a layer in your neural network. The goal is to build a tall tower (deep network) that can do a specific task, like recognizing cats in images.

1) Building the Tower:
As you add more blocks (layers) to your tower, it becomes harder to build it straight and reach the desired height (train effectively). This is because, in deep towers, the lower blocks might not get the right instructions from the top layers due to tiny voices (gradients) getting quieter as they go down.

2) Skip Connections - Bridges Between Blocks:
Now, let's add bridges (skip connections) between some of the blocks. These bridges let you directly connect lower blocks to higher blocks. So, if there's something important in the lower blocks that gets lost as you go up, you can still easily access it through the bridges.

3) Shortcut to the Top:
With these bridges, you're not only building up from the ground but also jumping up partway using the bridges. It's like having an elevator in your tower. Even if the top layers are giving unclear instructions, you can still bring in good ideas from the bottom layers using the bridges.

4) Easier Tower Building:
Now, building the tower becomes easier. If you add more layers (blocks) on top, the bridges make sure that the bottom layers still have a say in the tower's construction. This way, the top layers can focus on improving the details without worrying too much about the overall structure.

5) Benefits of Bridges:
* You can build a taller tower (deeper network) because the bridges help information flow better between layers.
* The tower gets stronger since lower blocks can directly contribute to the final result.
* If you're building in a messy area (complex data), the bridges can guide you back to a clearer path.

In short, skip connections (bridges) in neural networks help make training deep networks easier and more effective. They ensure that important information from lower layers can still influence the final outcome, even when you're working with very deep structures. This way, deep networks can learn better and solve complex tasks more accurately.
