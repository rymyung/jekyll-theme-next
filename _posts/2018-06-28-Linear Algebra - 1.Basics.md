---
title: 선형대수 - 1장 : 선형대수 기초
date: 2018-06-28 23:29:53
categories:
- Linear Algebra
- 선형대수
tags:
- Linear Algebra
- 선형대수
- Python
- 파이썬
---

# Chapter 1. Basic for Linear Algebra


```python
import numpy as np
```

## 1.1. Scalar

**Scalar** : a single number, e.g., \\(3.8 \\)


```python
scalar = 3.8
print("Example of scalar : ", scalar)
```

    Example of scalar :  3.8
    

## 1.2. Vector
**Vector** : an ordered list of numbers, e.g., $$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$


```python
vector = np.array([[1, 2, 3]])
print("Example of vector :", vector)
print("Shape of vector :", vector.shape)
```

    Example of vector : [[1 2 3]]
    Shape of vector : (1, 3)
    

* Column vector : a vector of *n*-dimension is usually a column vector, i.e., a matrix of the size \\(n \times 1 \\)  
$$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n = \mathbb{R}^{n\times1}$$
* Row vector : a row vector is usually written as its transpose, i.e.,  
$$x^T = {\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}}^T = \begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix} \in \mathbb{R}^{1\times n}$$  



```python
column_vector = np.array([[1], [2], [3]])
print("Column vector :\n", column_vector)
print("Dimension of column vector :", column_vector.shape)
```

    Column vector :
     [[1]
     [2]
     [3]]
    Dimension of column vector : (3, 1)
    


```python
print("Row vector :\n", column_vector.T)
print("Dimension of column vector :", column_vector.T.shape)
```

    Row vector :
     [[1 2 3]]
    Dimension of column vector : (1, 3)
    

## 1.3. Matrix
**Matrix** : a two-dimensional array of numbers

* \\(A \in \mathbb{R}^{n\times n} \\) : Square matrix (# of rows = # of columns)  
$$A = \begin{bmatrix} 1 & 6 \\ 3 & 4 \end{bmatrix}$$


```python
square_matrix = np.array([[1,6], [3,4]])
print("Example of square matrix :\n" ,square_matrix)
print("Dimension of square matrix :" ,square_matrix.shape)
```

    Example of square matrix :
     [[1 6]
     [3 4]]
    Dimension of square matrix : (2, 2)
    

* \\(A \in \mathbb{R}^{m\times n} \\) : Rectangular matrix (possible :\# of rows \\(\ne \\) \# of columns)  
$$A = \begin{bmatrix} 1 & 6 \\ 3 & 4 \\ 5 & 2 \end{bmatrix}$$


```python
rec_matrix = np.array([[1,6], [3,4], [5,2]])
print("Example of rectangular matrix :\n" ,rec_matrix)
print("Dimension of rectangular matrix :" ,rec_matrix.shape)
```

    Example of rectangular matrix :
     [[1 6]
     [3 4]
     [5 2]]
    Dimension of rectangular matrix : (3, 2)
    

* \\(A^T \\) : Transpose of matrix (mirroring across the main diagonal)  
$$A^T = \begin{bmatrix} 1 & 3 & 5 \\ 6 & 4 & 2 \end{bmatrix}$$


```python
tran_matrix = rec_matrix.transpose()
print("Example of transpose matrix :\n" ,tran_matrix)
print("Dimension of transpose matrix :" ,tran_matrix.shape)
```

    Example of transpose matrix :
     [[1 3 5]
     [6 4 2]]
    Dimension of transpose matrix : (2, 3)
    

* \\(A_{ij} \\) : (i, j)-th component of A, e.g. \\(A_{2,1} \\) = 3


```python
print(rec_matrix[1, 0])
```

    3
    

* \\(A_{i,} \\) : i-th row vector of A, e.g. \\(A_{2,} \\) = [3, 4]


```python
print(rec_matrix[1])
```

    [3 4]
    

* \\(A_{,j} \\) : j-th column vector of A, e.g. \\(A_{,2} \\) = [6,4,2]


```python
print(rec_matrix[:,1])
```

    [6 4 2]
    

## 1.4. Vector/Matrix Additions and Multiplications

* \\(C = A + B \\) : Element-wise addition, i.e., \\(C_{i,j} = A_{i,j} + B_{i,j} \\)
    * A, B should have the same size , i.e., \\(A, B \in \mathbb{R}^{mxn} \\)  
$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \\ 9 \end{bmatrix}$$


```python
A = np.array([1,2,3])
B = np.array([4,5,6])
print(A + B)
```

    [5 7 9]
    

* \\(ka, kA \\) : Scalar multiple of vector/matrix  
$$2 \times \begin{bmatrix} 3 \\ 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \\ 8 \end{bmatrix}$$  
$$2 \times \begin{bmatrix} 1 & 6 \\ 3 & 4 \\ 5 & 2 \end{bmatrix} = \begin{bmatrix} 2 & 12 \\ 6 & 8 \\ 10 & 4 \end{bmatrix}$$


```python
print(2*vector)
print(2*rec_matrix)
```

    [2 4 6]
    [[ 2 12]
     [ 6  8]
     [10  4]]
    

* \\(C = AB \\) : Matrix-matrix multiplication, i.e., \\(C_{i,j}  = \sum_k A_{i,k}B_{k,j} \\)  
$$\begin{bmatrix} 1 & 6 \\ 3 & 4 \\ 5 & 2 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 13 & 5 \\ 11 & 1 \\ 9 & -3 \end{bmatrix}$$  
$$size : (3 \times 2)(2 \times 2) = 3 \times 2 $$


```python
A = np.array([[1,6], [3,4], [5,2]])
B = np.array([[1,-1], [2,1]])
print(np.matmul(A,B)) # A.dot(B)
```

    [[13  5]
     [11  1]
     [ 9 -3]]
    

## 1.5. Properties

* \\(AB \ne BA \\) : ***not*** Commutative(교환법칙 성립 x)
    * e.g., Given \\(A \in \mathbb{R}^{2\times3} \\) and \\(B \in \mathbb{R}^{3\times5} \\), \\(AB \\) is defined, but \\(BA \\) is not even defined
    * Even though \\(BA \\) is defined, e.g., \\(A \in \mathbb{R}^{2\times3} \\) and \\(B \in \mathbb{R}^{3\times2} \\), still the sizes of \\(AB \in \mathbb{R}^{2\times2} \\) and \\(BA \in \mathbb{R}^{3\times3} \\) does not match, so \\(AB \ne BA \\)
    * Even though the sizes of \\(AB \\) and \\(BA \\) match, e.g., \\(A \in \mathbb{R}^{2\times2} \\) and \\(B \in \mathbb{R}^{2\times2} \\), still in this case generally \\(AB \ne BA \\)  
    $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$  
    $$\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}$$
* \\(A(B+C) = AB + BC \\) : Distributive(분배법칙 o)
* \\(A(BC) = (AB)C \\) : Associative(결합법칙 o)
* \\((AB)^T = A^TB^T \\) : Property of transpose
* \\((AB)^{-1} = A^{-1}B^{-1} \\) : Property of inverse


```python
print(A.dot(B))
```

    [[13  5]
     [11  1]
     [ 9 -3]]
    


```python
print(B.dot(A)) # Error
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-01141af5e94f> in <module>()
    ----> 1 print(B.dot(A)) # Error
    

    ValueError: shapes (2,2) and (3,2) not aligned: 2 (dim 1) != 3 (dim 0)

