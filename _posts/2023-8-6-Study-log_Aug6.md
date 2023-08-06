---
layout: post
title: Study log_Aug6
tags:
  - study log
  - Python
  - algorithms
  - Youtube channels
use_math: true
comments: true
---

## Programmers

This morning, I solved two problems with Programmers. 
The first one was a simple question where I had to determine whether the inputted number is odd or even. My initial solution was okay, but then I stumbled upon an incredibly clever and compact code snippet shared by another participant:
```python
n=int(input())

print(f"{n} is {'eovdedn'[n&1::2]}")
```

I must admit, I was amazed by how brilliantly efficient this code is compared to my own:

```python
a = int(input())
if a%2 == 0:
    print(a, "is even")
else: 
    print(a, "is odd")
```

Sometimes, seeing such elegant solutions can make me feel a bit down, but the beauty of programming is that people generously share their knowledge and code. 
It's a wonderful opportunity for me to learn and grow. 
So, I'll stay motivated and keep absorbing new techniques! 🌟

## Sololearn 

Also, I solved one Sololearn Python intermediate course. 
Learned about list comprehension. 
Code: 
```python
word = input()
vowels = "aeiou"
n_vowels = [char for char in word if char not in vowels]
print(n_vowels)
```

These days I tend to ask Chat GPT to improve my code after I make my own solution. 
It helps to learn new ways and also to make more concise and efficient code. 
Improved code by GPT: 
```python
word = input()
vowels = "aeiou"
n_vowels = list(filter(lambda char: char not in vowels, word))
print(n_vowels)
```
Chat GPT has been a fantastic coding buddy! 👩‍💻

## Youtube

I'm also watching a very interesting Youtube Channel named "생활코딩(Life Coding)". 
I found this channel while looking for some interesting and fun explanation about Machine Learning and this was exactly what I was looking for. 
Link: <https://youtu.be/LPqmPfhnR1o>

I highly recommend it!
