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

## Google Colab Initial Setting

In the class, we used google colab. 
The first thing we did was to change the runtime type to T4 GPU and write the following code to check the colab-GPU, RAM, and pytorch-GPU. 

```python
from psutil import virtual_memory
import torch

# colab-GPU 사용 확인
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('GPU 연결 실패!')
else:
  print(gpu_info)

# RAM 사용량 체크
ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))

# pytorch-GPU 연결 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('학습을 진행하는 기기:',device)

# 구글 드라이브 연결. 만약 직접 데이터셋을 사용한다면 주석 해제.
# from google.colab import drive
# drive.mount('/content/drive')
```

## Creating a custom dataset class in PyTorch using the Dataset and DataLoader classes
```python
import torch
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset): #데이터 전 처리를 진행해 주는 클래스
  def __init__(self, x, y): #constructor method: initialize the attributes and properties of the newly created instance.
                            #when defining a method within a class, including the '__init__' constructor, you need to include the 'self' parameter as the first parameter in the method's signature. This is because methods in Python classes are designed to operate on instances of the class, and the 'self' parameter serves as a reference to the instance itself.
                            #so, the 'self' parameter works a an instance context and attribute access. Without 'self', the method wouldn't know where to find the instance's attributes, leading to errors.
                            #lastly, in a single instance of a class, the self parameter refers to the same object throughout all of its methods
    self.x = x
    self.y = y

  def __getitem__(self, index): #This method is crucial for retrieving data from your custom dataset.데이터를 불러옴
    #숙제 할 땐 이 안에서 텐서화 시키기
    return self.x[index], self.y[index] #it returns a tuple containing self.x[index] and self.y[index], essentially providing one input-output pair.


  def __len__(self): #This method returns the length of the dataset. It's important for various operations, such as determining the number of batches in a DataLoader. 데이터 로더에 넣기 위해서 전체 길이가 얼마인지 알아야 함
    return self.x.shape[0] #returns the number of samples in self.x.
```

## Generating random data to be used in the dataset
```python
import numpy as np

x = np.random.randint(0, 100, 5)
y = np.random.randint(0, 100, 5)
print(x)
print(y)
x = torch.Tensor(x)
y = torch.Tensor(y)
print(x)
print(y)
dataset = myDataset(x, y)
print(dataset[0])
```

## DataLoader
```python
dataloader = DataLoader(dataset, batch_size=3, num_workers=2, pin_memory=True)
```

## Check Result
```python
x, y = next(iter(dataloader))#강제로 반복시킨다
print(x, y)
print(x.shape, y.shape)#어떻게 생겼는지 형태 확인
```

```python
#result
tensor([22., 66., 33.]) tensor([37., 59., 88.])
torch.Size([3]) torch.Size([3])
```
The output of the print(x, y) statements will show the actual values of your input data and labels in the batch, and the output of the print(x.shape, y.shape) statements will show the shapes of the tensors, indicating the number of samples in the batch and the dimensions of the individual samples.

## Homework
1. Create my own Dataset Class using PyTorch.
2. Load 10 image files
3. Convert RGB to Lab and Lab to RGB
4. Apply image processessing: Implement and apply various image processing operations to the images. This can include operations like resizing, cropping, flipping, rotating, or any other image transformation that you're interested in exploring.




