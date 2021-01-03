---
layout: post
title: (TIL) 파이썬- map 함수
tags: [python,TIL]
math: true
date: 2021-01-03 19:26 
comments : true
---

오늘은 파이썬의 map 함수에 대해 정리해보겠습니다. 

##  map 내장 함수 
---
`map()`은 반복 가능한 `iterable` 객체를 받아서, 각 요소에 함수를 적용해주는 함수이다. List나 tuple을 대상으로 주로 사용함

```python
target = [1,2,3,4]
```

각 target 값에 1을 더해주는 함수를 만든다고 가정한다면, <br>

```python
def add_value(n):
    return n+1 

target = [1,2,3,4]
result = [] 

for num in target:
    result.append(add_value(num))

```

> map 함수 >> map (적용 함수,적용 요소들) <br>

map 함수를 이용하면, 같은 연산을 더 쉽게 메모리도 절약할 수 있다.

```python
# map 함수 적용 
result = map(add_value,target)
print(list(result))

# 람다 함수로도 적용 가능 
result = map(lambda x:x+1,target)

```
