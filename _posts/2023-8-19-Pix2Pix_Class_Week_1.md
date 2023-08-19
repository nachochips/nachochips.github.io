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

### Loss Function 

In machine learning, a loss function (also known as a cost function or objective function) is a mathematical function that measures how well a machine learning model's predictions match the actual target values in the training data. The goal of a machine learning model is to minimize this loss function. In simpler terms, the loss function quantifies how "wrong" the model's predictions are.

<img src="/docs/assets/linearlog.png" width="80%" height="80%">
[Source: https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/]

In regression tasks (predicting continuous values), a common loss function is the Mean Squared Error (MSE). It calculates the average squared difference between predicted and actual values.
In classification tasks (categorizing into classes), the Cross-Entropy Loss (also called Log Loss) is often used. It measures the dissimilarity between predicted probabilities and the true class labels.

The process of training a machine learning model involves finding the parameters (weights and biases) that minimize the chosen loss function. Optimization algorithms, such as gradient descent, are used to adjust these parameters iteratively to reduce the loss.

Remember, the ultimate goal is to find the model's parameters that lead to the smallest loss value, which means the model's predictions are as close as possible to the actual target values.

#### Differentiation of the Loss Function 

