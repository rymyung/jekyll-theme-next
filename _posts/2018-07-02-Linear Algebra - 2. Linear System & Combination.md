---
title: Linear Algebra - Chapter 2. Linear System & Combination
date: 2018-07-02 15:41:53
categories:
- Linear Algebra
tags:
- Linear Algebra
- Python
---

이 포스트는 *인공지능을 위한 선형대수* 강의를 요약한 것입니다. [Link](https://www.edwith.org/linearalgebra4ai/joinLectures/14072)


```python
import numpy as np
```

# 선형 방정식 및 선형 시스템

## 선형 방정식(Linear Equation)
* \\(x_1, x_2, \cdots, x_n \\)을 변수로 가진 선형방정식은 다음과 같은 형태로 나타냄 $$ a_1x_1 + a_2x_2 + \cdots + a_nx_n = b$$
\\(b \\) 와 coefficients \\(a_1, a_2, \cdots, a_n \\)는 실수 혹은 복소수
* 위의 방정식은 다음과 같이 나타낼 수 있음  
$$\mathbf{a}^T\mathbf{x} = \mathbf{b}$$  
$$where,\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}, \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}.$$

## 선형 시스템(Linear System)
* 선형시스템은 선형방정식들의 집합(연립방정식)
* 선형시스템의 예 :
    
    
    1. 데이터    
PearsonID | Weight | Height | Is_smoking | Life-span
----------|--------|--------|------------|----------
1         |60kg    |5.5ft   |Yes (=1)    |66
2         |65kg    |5.0ft   |No (=0)     |74
3         |55kg    |6.0ft   |Yes (=1)    |78  

    2. 선형 시스템 set up
$$60x_1 + 5.5x_2 + 1 \cdot x_3 = 66$$ $$65x_1 + 5.0x_2 + 0 \cdot x_3 = 74$$ $$55x_1 + 6.0x_2 + 1 \cdot x_3 = 78$$

    3. 선형 시스템의  모든 coefficients를 matrix로 구성
$$\mathbf{A} = \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix}$$

    4. 타겟 vector와 가중치 vector 구성
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \mathbf{b} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$

    5. 하나의 matrix equation으로 변환
$$\begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$
$$\mathbf{A} \mathbf{x} = \mathbf{b}$$


## 단위 행렬/항등 행렬(Identity matrix)
* 주대각선이 전부 1이고 나머지 요소는 0을 값으로 갖는 \\(n \times n \\) 정방행렬(square matrix), \\(I_n = \mathbb{R}^n \\)
* e.g., \\(I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \\)
* \\(AI_n = I_nA = A \\)


```python
print(np.identity(3))
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    

## 역행렬(Inverse Matrix)
* 정방행렬(square matrix) \\(A \in \mathbb{R}^{n\times n} \\)에 대하여, 역행렬 \\(A^{-1} \\)은 다음과 같이 정의됨
$$A^{-1}A = AA^{-1} = I_n$$
* \\(2 \times 2 \\) 행렬 \\(A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \\)에 대하여, 역행렬 \\(A^{-1} \\)은
$$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$
* \\(A \in \mathbb{R}^{n\times n} \\) where \\(n >= 3 \\)의 역행렬은 가우스 소거법(Gaussian elimination)을 이용해 계산


```python
A = np.array([[2,3,-2], [3,5,6], [2,4,3]])
print(np.linalg.inv(A))
```

    [[ 0.69230769  1.30769231 -2.15384615]
     [-0.23076923 -0.76923077  1.38461538]
     [-0.15384615  0.15384615 -0.07692308]]
    

## 이용한 선형시스템의 해

### 역행렬이 존재하는 경우
* \\(A \\)의 역행렬이 존재하는 경우(\\(A \\)가 가역행렬(invertible matrix))
    * 해(solution)은 \\(\mathbf{x} = \mathbf{A}^{-1}\mathbf{b} \\)로 유일하게 존재
$$\mathbf{A}\mathbf{x} = \mathbf{b}$$
$$\mathbf{A}^{-1}\mathbf{A}\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$
$$I_n\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$
$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$


```python
A = np.array([[2,3,-2], [3,5,6], [2,4,3]])
b = np.array([[-5], [12], [7]])
Ainv = np.linalg.inv(A)
x = np.dot(Ainv, b)
print(x)
```

    [[-2.84615385]
     [ 1.61538462]
     [ 2.07692308]]
    


```python
print(np.linalg.solve(A, b))
```

    [[-2.84615385]
     [ 1.61538462]
     [ 2.07692308]]
    

### 역행렬이 존재하지 않는 경우
* \\(A \\)의 역행렬이 존재하지 않는 경우(\\(A \\)가 특이행렬(singular matrix))
    * \\(det A = 0(det A = ad-bc, A \in \mathbb{R}^{2\times2}) \\)인 경우, \\(A \\)의 역행렬이 존재하지 않음
    * \\(det A = 0 \\)이면, \\(\mathbf{A}\mathbf{x} = \mathbf{b} \\)의 해가 무수히 많거나(infinitely many solutions) 해가 없음(no solution)
    

### Rectangular Matrix의 경우
* \\(A \in \mathbb{R}^{m \times n} \\), \\(m < n \\) : variables이 equations보다 많은 경우, 일반적으로 해가 무수히 많음(under-determined system)
* \\(A \in \mathbb{R}^{m \times n} \\), \\(m > n \\) : variables이 equations보다 적은 경우, 일반적으로 해가 없음(over-determined system)
* 가장 근사적으로 만족시키는 해를 구하는 것은 가능(e.g. regularization, least squares)

# 선형 결합

## 선형 결합(Linear Combination)
선형결합이란, 주어진 벡터들 \\(v_1, v_2, \cdots, v_p \in \mathbb{R}^n \\)에 상수(스칼라들 \\(c_1, c_2, \cdots, c_p \\))배를 해서 더해주는 형태로, \\(c_1v_1 + c_2v_2 + \cdots + c_pv_p \\)를 \\(v_1, v_2, \cdots, v_p \\)의 선형결합이라고 한다.  
이 때 \\(c_1, c_2, \cdots, c_p \\)를 가중치(weight) 혹은 계수(coefficient)이라고 하고, 선형 결합의 가중치들은 일반적으로 0을 포함한 실수만을 다룬다.

선형 결합을 이용해서 linear system의 matrix equation을 vector equation으로 변환할 수가 있다.
$$\begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$
$$\mathbf{A} \mathbf{x} = \mathbf{b}$$
$$\Downarrow$$
$$\begin{bmatrix} 60 \\ 65 \\ 55 \end{bmatrix}x_1 + \begin{bmatrix} 5.5 \\ 5.0 \\ 6.0 \end{bmatrix}x_1 + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}x_1 = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$
$$a_1x_1 + a_2x_2 + a_3x_3 = b$$

## 생성(Span)
 \\(Span\{v_1, v_2, \cdots, v_p\} \\)이란, 벡터들 \\(v_1, v_2, \cdots, v_p \in \mathbb{R}^n \\)가 주어졌을 때, \\(v_1, v_2, \cdots, v_p \\)의 모든 선형 결합의 집합으로 정의된다.  
즉, \\(Span\{v_1, v_2, \cdots, v_p\} \\)은 임의의 scalars \\(c_1, c_2, \cdots, c_p \\)를 가지고 \\(c_1v_1 + c_2v_2 + \cdots + c_pv_p \\)로 표현될 수 있는 가능한 모든 벡터들의 집합이라고 할 수 있다.  
또한 \\(v_1, v_2, \cdots, v_p \\)에 의해 생성(\mathbb{R}^n \\)의 부분 집합이라고 할 수 있다.

### Span의 기하학적 의미
* 예시 :
![Span예시](https://www.dropbox.com/s/fgxicws9hjo16wl/span.jpg?raw=1)
    \\(v_1 = \begin{bmatrix} a \\ b \\ c \end{bmatrix} \\)와 \\(v_2 = \begin{bmatrix} x \\ y \\ z \end{bmatrix} \\)가 주어졌을 때, \\(v_1 \\)과 \\(v_2 \\)로 만들어질 수 있는 평면에 있는 모든 점들의 집합이 \\(Span\{v_1, v_2\} \\)이다.  
즉, \\(v_1, v_2 \in \mathbb{R}^n \\)가 non-zero 벡터이고 \\(v_2 \\)가 \\(v_1 \\)의 곱이 아닌 경우, \\(Span\{v_1, v_2\} \\)은 \\(v_1,v_2, 0 \\)을 가진 \\(\mathbb{R}^n \\) 상의 평면(\\(\mathbb{R}^n \\)의 부분 집합)이다.

## Vector Equation의 Solution
다음의 방정식에 해가 존재할까?
$$\begin{bmatrix} 60 \\ 65 \\ 55 \end{bmatrix}x_1 + \begin{bmatrix} 5.5 \\ 5.0 \\ 6.0 \end{bmatrix}x_1 + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}x_1 = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$
$$a_1x_1 + a_2x_2 + a_3x_3 = b$$

좌측의 식은 \\(a_1, a_2, a_3 \\) 벡터들의 선형 결합이 된다. 선형 결합의 계수 \\(x_1, x_2, x_3 \\)를 잘 조절해서 주어진 벡터 \\(b \\)를 만족하도록 하면, 계수 \\(x_1, x_2, x_3 \\)는 이 방정식의 해가 된다. 즉, \\(a_1, a_2, a_3 \\) 3개의 벡터에 의해 만들어진 \\(Span\{a_1, a_2, a_3\} \\) 안에 \\(b = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix} \\)가 포함되어 있다면, \\(b \\)를 \\(a_1, a_2, a_3 \\)의 선형 결합으로 나타낼 수 있다. 따라서 \\(b \in Span\{a_1, a_2, a_3\} \\)인 경우에 해가 존재한다.

### 방정식과 미지수의 관점
* 미지수(varible)의 갯수 : 주어진 벡터의 수, # of \\(\{v_1, v_2, \cdots \} \\)  
* 방정식(equation)의 갯수 : 주어진 벡터가 존재하는 전체 집합(공간)의 차원, \\(\mathbb{R}^n \\)  

방정식의 갯수가 많다는 것은, 주어진 벡터가 존재하는 전체 집합(공간)의 차원이 매우 크다는 것을 의미하고, 미지수의 갯수가 적다는 것은 Span을 하는데 사용되는 벡터들 \\(\{v_1, v_2, \cdots \} \\)은의 수가 적어 전체 집합(공간)의 차원에 비해서 몇 개 안되는 것을 의미한다.

$$ v_1 \cdot x_1 + v_2 \cdot x_2 = b $$
$$ v_1, v_2 \in \mathbb{R}^{10}$$
예를 들어, 10차원의 공간(\\(\mathbb{R}^{10} \\))에서 주어진 벡터들이 2개(\\(\{v_1, v_2\} \\))가 있다면, \\(Span\{v_1, v_2\} \\) 부분 공간은 전체 공간에 비해 매우 작은 공간이 된다. 그 때 \\(b \\)가 Span된 매우 작은 부분 공간 안에 속해야 해가 존재하는데, 미지수의 갯수가 방정식의 갯수보다 매우 적은 경우에는 해가 존재할 가능성이 매우 낮게 된다.

## 행렬의 곱셈
왼쪽 행렬의 row와 오른쪽 행렬의 column 간의 내적(inner product)로 행렬의 곱을 정의했었다.  
예를 들어,$$\begin{bmatrix} 1 & 6 \\ 3 & 4 \\ 5 & 2 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 13 & 5 \\ 11 & 1 \\ 9 & -3 \end{bmatrix}$$의 경우, 총 6번의 내적을 따로따로 해야한다.

이러한 방법 대신, 한 번에 계산할 수 있는 관점이 있다. \\(Ax \\)를 왼쪽 행렬의 column vector들의 선형 결합으로 보는 관점이다.
* matrix by vector : $$\begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 1 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2\\ 3 \end{bmatrix} = \begin{bmatrix} 1 \\ 1\\ 1 \end{bmatrix}1 + \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}2 + \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}3$$

* matrix by matrix : $$\begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 1 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 2 & 0 \\ 3 & 1 \end{bmatrix} = \begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ x_3 & y_3 \end{bmatrix} = \begin{bmatrix} x & y \end{bmatrix}$$ 오른쪽 첫번째 column vector(\\(\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \\))은 \\(y \\)에 영향을 미치지 않고 오른쪽 두번째 column vector(\\(\begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix} \\))는 \\(x \\)에 영향을 미치지 않는다.
따라서, 
$$x =  \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} =  \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}1 +  \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}2 +  \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}3$$ 
$$y =  \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} =  \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}(-1) +  \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} 0 +  \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}1$$
와 같이 나타낼 수 있다. 결국 \\(x \\)와 \\(y \\)는 왼쪽 행렬의 column vector들의 span에 무조건 포함되게 된다.

# 선형 독립과 선형 종속
\\(a_1x_1 + a_2x_2 + a_3x_3 = b \\)의 해가 존재할 때, \\(a_1, a_2, a_3 \\)가 선형 독립이라면 유일한 해가 존재하고 선형 종속이라면 해가 무수히 많이 존재한다.

## Practical Definition
\\(v_1, v_2, \cdots, v_p \in \mathbb{R}^n \\)가 주어졌을 때,  
\\(v_j \\)가 이전의 벡터들 \\(\{v_1, v_2, \cdots, v_{j-1}\} \text{ for }j = 1, \cdots, p \\)의 선형 결합으로 표현 가능한지를 확인하여
* 적어도 하나의 \\(v_j \\)가 있다면 \\(\{v_1, v_2, \cdots, v_p\} \\)는 선형 종속,
* \\(v_j \\)가 하나도 없다면 \\(\{v_1, v_2, \cdots, v_p\} \\)는 선형 독립이다.
$$\{v_1, v_2, \cdots, v_p \} \begin{cases} \text{linear dependence}, \text{ if } v_j \in Span\{v_1, v_2, \cdots, v_p\} \text{ for some } j = 1, \cdots, p \\ \text{Linear Independence}, \text{ if } v_j \notin Span\{v_1, v_2, \cdots, v_p\} \text{ for some } j = 1, \cdots, p \end{cases}$$

선형 종속인 벡터는 span을 증가시킬 수 없다.
$$\text{If } v_3 \in Span\{v_1, v_2\} \rightarrow Span\{v_1, v_2\} = Span\{v_1, v_2, v_3\}$$

## Formal Definition
\\(x_1v_1 + x_1v_2 + \cdots + x_pv_p = \mathbf{0} \\)를 고려해보자. 이 경우에 \\(\mathbf{0} \\) 벡터는 항상 \\(Span\{v_1, v_2, \cdots, v_p\} \\)에 무조건 포함되어 있다.  
즉, 이 방정식은 최소한 하나의 해(\\(x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \\), trivial solution)를 가지고 있다.  
* 이 trivial solution이 유일한 해이면, \\(v_1, v_2, \cdots, v_p \\)는 선형 독립,  
* 이 trivial solution이 유일한 해가 아니면, \\(v_1, v_2, \cdots, v_p \\)는 선형 종속이다.
