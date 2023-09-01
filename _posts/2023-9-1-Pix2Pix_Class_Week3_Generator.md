---
layout: post
title: Pix2Pix Class Week3_Generator
tags:
  - studylog
  - Python
  - Pytorch
  - Pix2Pix
  - Deeplearning
use_math: true
comments: true
---
## DataLoader Homework review

First, download images to be used in the DataLoader from COCO_SAMPLE

```python
#아마 앞으로 다시는 쓸 일 없을 코드. 왜냐하면 사진을 쉽게 더 빠르게 다운받기 위해 쓰는 것
from fastai.data.external import untar_data, URLs
#다만 연구를 할 때는 매 번 이렇게 다운 받아서 사용하는 것은 불안정하고, 사진 구성을 통제할 수 없음. 쓸 거면 다운로드 받아서 사용 필요.
coco_path = untar_data(URLs.COCO_SAMPLE)
```

Now, check where the images have been downloaded. 

```python
coco_path #다운로드 된 사진 파일 위치

```
Find all the .jpg image files in the directory specified by str(coco_path) + '/train_sample/'.

```python
import glob #glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
paths = glob.glob(str(coco_path) + '/train_sample/*.jpg') #*을 씀으로 모든 jpg파일을 부름
```
Count the total number of images
```python
len(paths) #이미지 수 세기
```

Next, we will randomly select 10,000 images from the sample images and divide them to be 8,000 training images and 2,000 validation images. 

```python
import numpy as np
#In np.random.seed(1), 1 is the seed for this pseudorandom number generation algorithm.
#If you have the same seed value, then the pseudorandom numbers generated afterwards will always follow the same pattern.
#Therefore, if you set your seed value using np.random.seed(1), you'll get identical results each time your code runs.
#This property is often used to ensure reproducibility in results when performing data analysis or training machine learning models
np.random.seed(1)
#Randomly selects 10,000 elements from the list paths.
#The parameter replace=False means that once an element has been chosen it cannot be selected again (i.e., no element will appear more than once in chosen_paths).
chosen_paths = np.random.choice(paths, 10000, replace = False)
index = np.random.permutation(10000) #10000 안에서 랜덤으로 인덱스 생성

train_paths = chosen_paths[index[:8000]] #8000번째 까지를 training set 경로로 지정
val_paths = chosen_paths[index[8000:]]
```
To visualize, let's plot an image. 
We can see that a image will show up successfully. 

```python
import matplotlib.pyplot as plt
import matplotlib #상당히 비안정적인 라이브러리라 특히 코랩에서 할 때는 두 개 안 쓰면 안 되는 경우가 있음

img = matplotlib.image.imread(train_paths[0])
plt.imshow(img)
plt.axis("off")
plt.show()
```
Finally the main DataLoader code: 
```python
#데이터 전처리하기
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
class ColorizationDataset(Dataset):
  def __init__(self, paths, mode = 'train'):
    self.paths = paths
    self.mode = mode

    if mode == 'train':
      self.transforms = transforms.Compose([ #Compose를 쓰면 다양한 기법을 한 번에 적용 가능
          transforms.Resize((256, 256), Image.BICUBIC), #사이즈 통일 및 어떤 알고리즘으로 resize할건지 선언(pil은 이미지 축소 알고리즘)
          transforms.RandomHorizontalFlip() #수평 반전으로 데이터 증강
      ])
    elif mode == 'validation': #굳이 증강할 필요 없이 원본 이미지로 검증하면 됨
      self.transforms = transforms.Resize((256, 256), Image.BICUBIC) #굳이 컴포즈 사용할 필요 없음
    else:
      raise Exception("Train or Validation Only!!")

  def __getitem__(self, index): #이미지가 불려질 때마다 사용되는 함수. 즉 어떻게 이미지를 불러올 것이냐
    img = Image.open(self.paths[index]).convert("RGB") #생성자에서 저장해둔 paths. 흑백 이미지의 경우 채널이 하나만 있어 오류가 있을 수 있기 때문에 rgb 단서를 꼭 넣어줌
    img = np.array(self.transforms(img)) #이미지를 열은 뒤 전처리를 시행 해줌 + Lab color space로 바꿔주기 위해 numpy array 활용
    img = rgb2lab(img).astype("float32") #pytorch tensor가 float32를 좋아함. fool proof 하는 데이터 표현 방법
    img = transforms.Tensor()(img) #이제 Tensor 형태로 바꿔주기
    #L채널 ab 채널 따로 return
    L = img[[0], ...] / 50. - 1.  #'...'은 그리고 모든 것을 의미함
    ab = im[[1, 2], ...] / 110.   # 논문 저자들이 이런 일(뒷단의 나눠주기 행위)을 한 이유는 0-255 범위가 넘 넓어서 -1 ~ 1사이로 정규화 하는 작업. 즉 이렇게 하면 각 채널 전체 scope가 -1 ~ 1로 변환. 꼭 안 해도 되지만 (netwrok가 sturdy해서) 저자가 햇으니깐 ㅎㅎ

    return {'L': L, 'ab': ab}

  def __len__(self):
    return len(self.paths)
```
