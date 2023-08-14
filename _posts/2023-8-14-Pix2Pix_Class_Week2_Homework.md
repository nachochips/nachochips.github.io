---
layout: post
title: Pix2Pix Class Week2_Pytorch Code
tags:
  - studylog
  - Python
  - Pytorch
  - Pix2Pix
  - Deeplearning
use_math: true
comments: true
---

## Homework Instructions
1) Create my own Dataset Class using PyTorch.
2) Load 10 image files
3) Convert RGB to Lab and Lab to RGB
4) Apply image processessing: Implement and apply various image processing operations to the images. This can include operations like resizing, cropping, flipping, rotating, or any other image transformation that you’re interested in exploring.

### Necessary Libraries 
```python
import os #file operations
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
```

### Create my own Dataset Class 
```python
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None): #Setting transform to None as a default value in the constructor of the CustomDataset class allows for flexibility when creating instances of the class. By providing a default value of None, you're allowing users of the class to either specify a custom transformation pipeline when creating an instance or to use the dataset without any transformation if they prefer.
        self.image_folder = image_folder
        self.image_paths = os.listdir(image_folder)
        self.transform = transform

    def __len__(self): #returns the length of the dataset, which is the number of images
        return len(self.image_paths)

    def __getitem__(self, idx): #where the data loading and transformations happen
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
```

### Transform image
```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor()
])
```
Although I tried to put the transform definition within the __getitem__ function, 'NameError: name 'transform' is not defined' kept occuring as it was defined outside the scope of the class. 
So, I had to seperate the code. But I think it has better code organization as it enhances efficiency, reusability and debugging. 

### Dataset
```python
image_folder = '/content/drive/MyDrive/Colab Notebooks/image'
dataset = CustomDataset(image_folder=image_folder, transform=transform)
```

### DataLoader 
```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
```

### Visualize images
```python
# Iterate through batches and visualize images
for batch in dataloader:
    # 'batch' is a tensor of shape (batch_size, channels, height, width)

    # Convert the tensor back to images
    batch_images = batch.numpy()  # Convert tensor to numpy array
    batch_images = np.transpose(batch_images, (0, 2, 3, 1))  # Transpose to (batch_size, height, width, channels)

    # Display images from the current batch
    for image in batch_images: #plt is necessary of visual representation of the images during the execution of the code 
        plt.imshow(image)
        plt.axis('off')
        plt.show()
```

### Full Code 
```python
import os #file operations
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None): #Setting transform to None as a default value in the constructor of the CustomDataset class allows for flexibility when creating instances of the class. By providing a default value of None, you're allowing users of the class to either specify a custom transformation pipeline when creating an instance or to use the dataset without any transformation if they prefer.
        self.image_folder = image_folder
        self.image_paths = os.listdir(image_folder)
        self.transform = transform

    def __len__(self): #returns the length of the dataset, which is the number of images
        return len(self.image_paths)

    def __getitem__(self, idx): #where the data loading and transformations happen
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor()
])

image_folder = '/content/drive/MyDrive/Colab Notebooks/image'
dataset = CustomDataset(image_folder=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

# Iterate through batches and visualize images
for batch in dataloader:
    # 'batch' is a tensor of shape (batch_size, channels, height, width)

    # Convert the tensor back to images
    batch_images = batch.numpy()  # Convert tensor to numpy array
    batch_images = np.transpose(batch_images, (0, 2, 3, 1))  # Transpose to (batch_size, height, width, channels)

    # Display images from the current batch
    for image in batch_images: #plt is necessary of visual representation of the images during the execution of the code 
        plt.imshow(image)
        plt.axis('off')
        plt.show()
```

### Results 
![image](https://github.com/nachochips/nachochips.github.io/assets/68362149/4c99d195-3b45-4feb-b3f0-4d1e912e22ba)

Practicing with my baby's image gives me motivations 🥰👶 

## Practicing RGB to Lab 
The LAB color space is a color representation that separates color information into three components: L (lightness), A (green to red), and B (blue to yellow). It's designed to approximate human perception of color in a more uniform and perceptually meaningful way compared to other color spaces like RGB or CMYK. The LAB color space is often used in fields such as color science, image processing, and computer vision.

Here's a breakdown of the LAB color space components:

1. **L (Lightness)**: The L channel represents the brightness or lightness of the color. It ranges from 0 (black) to 100 (white). A higher L value corresponds to a lighter color, while a lower L value corresponds to a darker color.

2. **A (Green to Red)**: The A channel represents the color along the green-to-red axis. It ranges from negative values (green) to positive values (red). A negative A value represents a green color, and a positive A value represents a red color.

3. **B (Blue to Yellow)**: The B channel represents the color along the blue-to-yellow axis. It also ranges from negative values (blue) to positive values (yellow). A negative B value represents a blue color, and a positive B value represents a yellow color.

The LAB color space has several advantages:

* Perceptual Uniformity: The LAB color space is designed to be perceptually uniform, meaning that a certain amount of numerical change in each channel corresponds to a roughly equal perceptual change in color.

* Color Similarity: The Euclidean distance between two colors in the LAB color space is a good indicator of their perceived similarity.

* Decoupling of Lightness and Color: Unlike RGB, where changing the color may also affect the brightness, LAB allows you to manipulate color and brightness independently.

* Color Correction and Manipulation: LAB is often used for color correction and image manipulation because it separates color and lightness information.

Although I wanted to include the rgb2lab and lab2rgb code in the above code, it kept occurring errors. 
So, I tried to make a practice code seperately as below. 

```python
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt

# Load image 
im = imread("/content/baby_e/KakaoTalk_20230812_001148362_07.jpg")

# rgb2lab 
im_lab = rgb2lab(im.copy())  # Create a copy to keep the original RGB image

# Set the color channel values (the second and third channels) to zeros: 
im_lab[..., 1] = im_lab[..., 2] = 0

# Lab2rgb 
im_gray = lab2rgb(im_lab)

# Show the images
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Original image', size=15)
plt.subplot(122), plt.imshow(im_gray), plt.axis('off'), plt.title('Gray scale image', size=15)
plt.show()
```
### Results
![image](https://github.com/nachochips/nachochips.github.io/assets/68362149/02dda74b-4169-4a5c-95bc-5b6dcb028f6d)

