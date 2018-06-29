---
title: Python for ML - Chapter 1. Pythonic Code
date: 2018-06-28 23:29:53
categories:
- Python for ML
tags:
- Programming
- Python
---

# Chapter 1. Pythonic Code

## 1. Split & Join
* Split
* Join
* String
* List
* Unpacking

### 1.1 Split
String Type의 값을 나눠서 List 형태로 반환


```python
items = 'zero one two three'.split()
print(items)
```

    ['zero', 'one', 'two', 'three']
    


```python
example = 'python,jquery,javascript'
print(example.split(","))
```

    ['python', 'jquery', 'javascript']
    


```python
first, second, third = example.split(",")
print("first : {}, \nsecond : {}, \nthird : {}".format(a,b,c))
```

    first : python, 
    second : jquery, 
    third : javascript
    

### 1.2 Join


```python
colors = ['red', 'blue', 'green', 'yellow']
result = ', '.join(colors)
print(result)
```

    red, blue, green, yellow
    

## 2. List Comprehension
기존 List를 사용해서 간단한 다른 List를 만드는 기법
* List comprehension
* Nested for loop


```python
result = [i for i in range(10)]
print(result)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    


```python
result2 = [i for i in range(10) if i % 2 == 0]
print(result2)
```

    [0, 2, 4, 6, 8]
    

#### Nested for loop


```python
case1 = ['A', 'B', 'C']
case2 = ['D', 'E', 'A']
result3 = [i+j for i in case1 for j in case2 if not(i==j)]
print(result3)
```

    ['AD', 'AE', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA']
    


```python
words = 'The quick brown fox jumps over the lazy dog'.split()
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
stuff
```




    [['THE', 'the', 3],
     ['QUICK', 'quick', 5],
     ['BROWN', 'brown', 5],
     ['FOX', 'fox', 3],
     ['JUMPS', 'jumps', 5],
     ['OVER', 'over', 4],
     ['THE', 'the', 3],
     ['LAZY', 'lazy', 4],
     ['DOG', 'dog', 3]]



## 3. Enumerate & Zip
* Zip
* Enumerate

### 3.1 Enumerate
List의 element를 추출할 때 번호를 붙여서 추출


```python
for i, v in enumerate(['A', 'B', 'C']) :
    print(i, v)
```

    0 A
    1 B
    2 C
    


```python
mylist = ['a', 'b', 'c', 'd', 'e']
list(enumerate(mylist))
```




    [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]




```python
{i:j for i,j in enumerate('Hello Python World !'.split())}
```




    {0: 'Hello', 1: 'Python', 2: 'World', 3: '!'}



### 3.2 Zip
두 개의 List의 값을 병렬적으로 추출


```python
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for a, b in zip(alist, blist) :
    print(a, b)
```

    a1 b1
    a2 b2
    a3 b3
    


```python
a, b, c = zip((1,2,3), (10, 20, 30), (100, 200, 300))
print("a : {}, \nb : {}, \nc : {}".format(a,b,c))
```

    a : (1, 10, 100), 
    b : (2, 20, 200), 
    c : (3, 30, 300)
    


```python
[sum(x) for x in zip((1,2,3), (10,20,30), (100, 200, 300))]
```




    [111, 222, 333]



## 4. Lambda & MapReduce
* Lambda
* Map Function
* Reduce Function

### 4.1 Lambda
함수 이름 없이, 함수처럼 사용할 수 있는 익명함수
<code>lambda arg1, arg2, ... : expression</code>


```python
# General function
def f(x, y) :
    return x + y
print(f(1,4))
```

    5
    


```python
# Lambda function
f2 = lambda x, y : x + y
print(f2(1,4))
```

    5
    


```python
print((lambda x : x**2)(5))
```

    25
    

### 4.2 Map function
Sequence 자료형(list, tuple, ...) 각 element에 동일한 function을 적용
<code> map(function, list1, list2, ...) </code> map object로 반환하기 때문에 반환값에 list()를 사용


```python
ex = [1,2,3,4,5]
f = lambda x : x ** 2
print(list(map(f, ex)))
```

    [1, 4, 9, 16, 25]
    


```python
ex = [1,2,3,4,5]
f = lambda x, y : x + y
print(list(map(f, ex, ex)))
```

    [2, 4, 6, 8, 10]
    


```python
print(list(map(lambda x : x ** 2 if x % 2 ==0 else x, ex)))
```

    [1, 4, 3, 16, 5]
    

### 4.3 Reduce function
map function과 달리 list에 똑같은 함수를 적용해서 통합


```python
from functools import reduce
print(reduce(lambda x, y : x + y, [1,2,3,4,5]))
```

    15
    

## 5. Asterisk
* Asterisk  
\* 를 의미  
단순 곱셈, 제곱연산, 가변 인자 활용 등 다양하게 사용

### 5.1 \* args
몇개의 인자가 들어올지 모를 때 사용


```python
def asterisk_test(a, *args) :
    print(a, args)
    print(type(args))
    
asterisk_test(1,2,3,4,5,6)
```

    1 (2, 3, 4, 5, 6)
    <class 'tuple'>
    

### 5.2 \*\*kargs
키워드 인자를 넣어줄 때 사용(dict type으로 적용)


```python
def asterisk_test2(a, **kargs) :
    print(a, kargs)
    print(type(kargs))

asterisk_test2(1, b=2, c=3, d=4, e=5, f=6)
```

    1 {'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    <class 'dict'>
    

## 6. Data Structure
* Collections
* Data Structure
* deque
* Counter
* orderedDict
* defaultdict
* namedtuple

### 6.1 Collections
List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)  
편의성, 실행 효율 등을 사용자에게 제공  
아래의 모듈이 존재
* from collections import deque
* from collections import Counter
* from collections import orderedDict
* from collections import defaultdict
* from collections import namedtuple

### 6.2 deque
Stack과 Queue를 지원하는 모듈
List에 비해 효율적인 자료 저장 방식을 지원  
효율적 메모리 구조로 처리 속도 향상


```python
from collections import deque
deque_list = deque()
for i in range(5) :
    deque_list.append(i)
print(deque_list)
```

    deque([0, 1, 2, 3, 4])
    


```python
deque_list.appendleft(10)
print(deque_list)
```

    deque([10, 0, 1, 2, 3, 4])
    

rotate, reverse 등 Linked List의 특성을 지원 
기존 list 형태의 함수를 모두 지원


```python
deque_list.rotate(2)
print(deque_list)
```

    deque([3, 4, 10, 0, 1, 2])
    


```python
deque_list.extend([5,6,7])
print(deque_list)
```

    deque([3, 4, 10, 0, 1, 2, 5, 6, 7])
    


```python
deque_list.extendleft([5,6,7])
print(deque_list)
```

    deque([7, 6, 5, 3, 4, 10, 0, 1, 2, 5, 6, 7])
    


```python
print(deque(reversed(deque_list)))
```

    deque([7, 6, 5, 2, 1, 0, 10, 4, 3, 5, 6, 7])
    

### 6.3 orderedDict
Dict와 달리 데이터를 입력한 순서대로 dict를 반환


```python
from collections import OrderedDict
d2 = OrderedDict()
d2['x'] = 100
d2['y'] = 200
d2['z'] = 300
d2['l'] = 500
for k, v in d2.items() :
    print(k, v)
```

    x 100
    y 200
    z 300
    l 500
    

### 6.4 defaultdict
Dict type의 값에 기본 값을 지정, 신규값 생성 시 사용하는 방법


```python
d = dict()
print(d["first"]) # Error
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-39-e2657e0b96e1> in <module>()
          1 d = dict()
    ----> 2 print(d["first"]) # Error
    

    KeyError: 'first'



```python
from collections import defaultdict
d = defaultdict(object) # Default dictionary를 생성
d = defaultdict(lambda : 0) # Default 값을 0으로 설정
print(d["first"])
```

    0
    

### 6.5 Counter
Sequence type의 data elements의 갯수를 dict 형태로 반환


```python
from collections import Counter
#c = Counter()
c = Counter("gallahad")
print(c)
```

    Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})
    

### 6.6 namedtuple
Tuple 형태로 Data 구조체를 저장하는 방법  
저장되는 data의 variable을 사전에 지정해서 저장


```python
from collections import namedtuple
Point = namedtuple("Point", ['x', 'y'])
p = Point(11, y=22)
print(p[0] + p[1])
```

    33
    
