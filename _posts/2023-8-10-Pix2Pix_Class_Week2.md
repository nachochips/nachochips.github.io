---
layout: post
title: Pix2Pix Class Week2
tags:
  - studylog
  - Python
  - Pytorch
  - Pix2Pix
  - Metacode
  - Deeplearning
use_math: true
comments: true
---

I had my second class on Metacode regarding Pix2Pix. 
Today we learned about Image Data, Input Data, and DataLoader. 

### Image as Data

Pooling is another important operation commonly used in image processing and convolutional neural networks (CNNs) for feature extraction and dimensionality reduction. 
There are two main types of pooling:
| Type    | Features    |
| :-------- | :------------ |
| Max Pooling | Not only does it reduces the spatial resolution of the input(image) but also helps retain the most dominant features in a region(i.e. edges or textures) while discarding less relevant information. So, if a feature moves slightly within a region, the pooled value remains the same |
| Average Pooling | Average pooling is similar to max pooling, but instead of taking the maximum value from each region, it computes the average value. |
