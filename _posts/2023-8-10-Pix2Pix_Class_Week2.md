---
layout: post
title: Pix2Pix Class Week2
tags:
  - studylog
  - Python
  - Pytorch
  - Pix2Pix
  - Deeplearning
use_math: true
comments: true
---
**Class file**: [Week 2](https://nachochips.github.io/docs/assets/Week_2%20(1).pdf)

I had my second class on Metacode regarding Pix2Pix. 
Today we learned about Image Data, Input Data, and DataLoader. 

## Image as Data

Pooling is another important operation commonly used in image processing and convolutional neural networks (CNNs) for feature extraction and dimensionality reduction. 
There are two main types of pooling:

Pooling is another important operation commonly used in image processing and convolutional neural networks (CNNs) for feature extraction and dimensionality reduction. There are two main types of pooling:

| Type          | Features                                                                                                                             |
| :------------ | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Max Pooling**   | Not only does it reduce the spatial resolution of the input (image), but it also helps retain the most dominant features in a region (i.e., edges or textures) while discarding less relevant information. So, if a feature moves slightly within a region, the pooled value remains the same. |
| **Average Pooling** | Average pooling is similar to max pooling, but instead of taking the maximum value from each region, it computes the average value.      |

## DataLoader
### What is DataLoader and why do we use it?
**DataLoader is a utility provided by libraries like PyTorch that helps manage and efficiently load datasets during training and evaluation of models. It is particularly useful when dealing with large datasets that can't fit entirely in memory.** 

#### Parameters of DataLoader
![image](https://github.com/nachochips/nachochips.github.io/assets/68362149/bf52dea9-e8f2-45ef-99fb-b852e7f758dd)

Although there are several parameterrs in the library, the important parameters are as follows: 

1) dataset: This parameter specifies the dataset that the DataLoader will work with. It typically needs to be a 'tensor' format. In my project, the dataset is going to be image files.

2) batch_size: Determines the number of samples in each mini-batch. Training models with mini-batches allows for efficient memory utilization and can help regularize the learning process. Usually it's set to a mutiples of two (8, 16, 32). 

3) shuffle: If set to **True**, the dataset will be shuffled before each epoch. Shuffling helps break any inherent patterns in the data, leading to improved model generalization.

4) num_workers: Multi processing. Although it's not an answer or have concrete evidence, the num_workers is usually set to the 'number of GPU * 2' or 'number of GPU * 4'. Significally speed up data loading. 

5) pin_memory: Can be compared to making a reservation at a restaurant before you actually visit. When using a GPU for training, setting pin_memory to **True** can accelerate data transfer between the CPU and GPU.

6) drop_last: If set to **True**, the last mini-batch might be dropped if its size is smaller than batch_size. If you have enough large dataset you can set as **True**, however, in most cases it's safer to set to **False**. This ensures that all available data is utilized for training, even if the last mini-batch is smaller than the batch size. 

#### Workflow when using a 'DataLoader':
1) Dataset Creation: You create a custom dataset class that inherits from a library-specific dataset class (e.g., torch.utils.data.Dataset in PyTorch). This class defines how your data is loaded, transformed, and preprocessed.

2) Data Transformation: You define data transformation pipelines using libraries like torchvision (for images) or transforms (for general data).

3) DataLoader Creation: You create a DataLoader instance by passing your dataset and specifying batch size, shuffling, and other parameters.

4) Iteration: During training or evaluation, you iterate through the DataLoader. It loads data in mini-batches, applying transformations and collations as specified.

In the next post, I will write the first sample code for Pytorch DataLoader. 
