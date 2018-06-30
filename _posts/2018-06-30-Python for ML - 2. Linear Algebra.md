---
title: Linear Algebra
date: 2018-06-30 23:12:00
categories:
- Lecture
- Python for ML
- Programming
tags:
- python
---

간단한 선형대수의 수식들을 python code로 작성

# Problem 1.
vector 간 덧셈 또는 뺄셈을 연산할 때, 연산이 가능한 사이즈인지를 확인하여 가능 여부를 True 또는 False로 반환


```python
def vector_size_check(*vector_variables) :
    size_list = [len(i) for i in vector_variables]
    result = len(set(size_list))
    if result == 1 :
        return True
    else :
        return False
```


```python
print(vector_size_check([1,2,3], [2,3,4], [5,6,7])) # Expect value : True
print(vector_size_check([1,3], [2,4], [6,7])) # Expect value : True
print(vector_size_check([1,3,4], [4], [6,7])) # Expect value : False
```

    True
    True
    False
    

# Problem 2.
vector 간 덧셈을  실행하여 결과를 반환, 단 입력되는 vector의 갯수와 크기는 일정하지 않음   
$$\begin{bmatrix} a \\ b \\ c \end{bmatrix} + \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} a + x \\ b + y \\ c + z \end{bmatrix}$$


```python
def vector_addition(*vector_variables) :
    if vector_size_check(*vector_variables) == False :
        return "Error"
    return [sum(i) for i in zip(*vector_variables)]
```


```python
print(vector_addition([1, 3], [2, 4], [6, 7])) # Expected value: [9, 14]
print(vector_addition([1, 5], [10, 4], [4, 7])) # Expected value: [15, 16]
print(vector_addition([1, 3, 4], [4], [6,7])) # Expected value: Error
```

    [9, 14]
    [15, 16]
    Error
    

# Problem 3.
vector 간 뺄셈을 실행하여 결과를 반환, 단 입력되는 vector의 갯수와 크기는 일정하지 않음
$$\begin{bmatrix} a \\ b \\ c \end{bmatrix} - \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} a - x \\ b - y \\ c - z \end{bmatrix}$$


```python
def vector_subtraction(*vector_variables) :
    if vector_size_check(*vector_variables) == False :
        return "Error"
    return [x - y for x, y in zip(vector_variables[0], vector_addition(*vector_variables[1:]))]
```


```python
print(vector_subtraction([1, 3], [2, 4])) # Expected value: [-1, -1]
print(vector_subtraction([1, 5], [10, 4], [4, 7])) # Expected value: [-13, -6]
```

    [-1, -1]
    [-13, -6]
    

# Problem 4.
하나의 scalar 값을 vector에 곱함, 단 입력되는 vector의 크기는 일정하지 않음
$$\alpha \times \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} \alpha \times x \\ \alpha \times y \\ \alpha \times z \end{bmatrix}$$


```python
def scalar_vector_product(alpha, vector_variable) :
    return [alpha*i for i in vector_variable]
```


```python
print (scalar_vector_product(5,[1,2,3])) # Expected value: [5, 10, 15]
print (scalar_vector_product(3,[2,2])) # Expected value: [6, 6]
print (scalar_vector_product(4,[1])) # Expected value: [4]
```

    [5, 10, 15]
    [6, 6]
    [4]
    

# Problem 5.
matrix 간 덧셈 또는 뺄셈 연산을 할 때, 연산이 가능한 사이즈인지를 확인하여 가능 여부를 True 또는 False로 반환


```python
def matrix_size_check(*matrix_variables):
    return all([len(set(len(matrix[0]) for matrix in matrix_variables)) == 1]) and all([len(matrix_variables[0]) == len(matrix) for matrix in matrix_variables])
```


```python
matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
matrix_w = [[2, 5], [1, 1], [2, 2]]

print (matrix_size_check(matrix_x, matrix_y, matrix_z)) # Expected value: False
print (matrix_size_check(matrix_y, matrix_z)) # Expected value: True
print (matrix_size_check(matrix_x, matrix_w)) # Expected value: True
```

    False
    True
    True
    

# Problem 6.
비교가 되는 n개의 matrix가 서로 동치인지 확인하여 True 또는 False를 반환
$$\text{if } x = a, y = b, z = c, w = d \text{ then}$$
$$\begin{bmatrix} x & y \\ z & w \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$


```python
def is_matrix_equal(*matrix_variables) :
    return all([all(len(set(row)) == 1 for row in zip(*matrix)) for matrix in zip(*matrix_variables)])
```


```python
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]

print (is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y)) # Expected value: False
print (is_matrix_equal(matrix_x, matrix_x)) # Expected value: True
```

    False
    True
    

# Problem 7.
matrix 간 덧셈을 실행하여 결과를 반환, 단 입력되는 matrix의 갯수와 크기는 일정하지 않음
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} x & y \\ z & w \end{bmatrix} = \begin{bmatrix} a + x & b + y \\ c + z & d + w \end{bmatrix}$$


```python
def matrix_addition(*matrix_variables) :
    if matrix_size_check(*matrix_variables) == False :
        return "Different size"
    return [[sum(i) for i in zip(*row)] for row in zip(*matrix_variables)]
```


```python
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]

print (matrix_addition(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]
print (matrix_addition(matrix_x, matrix_y, matrix_z)) # Expected value: [[6, 11], [9, 6]]
```

    [[4, 7], [4, 3]]
    [[6, 11], [9, 6]]
    

# Problem 8.
matrix 간 뺄셈을 실행하여 결과를 반환, 단 입력되는 matrix의 갯수와 크기는 일정하지 않음
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} - \begin{bmatrix} x & y \\ z & w \end{bmatrix} = \begin{bmatrix} a - x & b - y \\ c - z & d - w \end{bmatrix}$$


```python
def matrix_subtraction(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    return [[x-y for x,y in zip(*row)] for row in zip(matrix_variables[0], matrix_addition(*matrix_variables[1:]))]
```


```python
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]

print (matrix_addition(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]
print (matrix_addition(matrix_x, matrix_y, matrix_z)) # Expected value: [[6, 11], [9, 6]]
```

    [[4, 7], [4, 3]]
    [[6, 11], [9, 6]]
    

# Problem 9.
matrix의 전치행렬을 구하여 결과를 반환함, 단 입력되는 matrix의 크기는 일정하지 않음
$$A = \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}, \text{ then } A^T = \begin{bmatrix} a & c & e \\ b & d & f \end{bmatrix}$$


```python
def matrix_transpose(matrix_variable):
    return [[element for element in row] for row in zip(*matrix_variable)]
```


```python
matrix_w = [[2, 5], [1, 1], [2, 2]]
matrix_transpose(matrix_w)
```




    [[2, 1, 2], [5, 1, 2]]



# Problem 10.
하나의 scalar 값을 matrix에 곱함, 단 입력되는 matrix의 크기는 일정하지 않음
$$\alpha \times \begin{bmatrix} a & c & d \\ e & f & g \end{bmatrix} = \begin{bmatrix} \alpha \times a & \alpha \times c & \alpha \times d \\ \alpha \times e & \alpha \times f & \alpha \times g \end{bmatrix}$$


```python
def scalar_matrix_product(alpha, matrix_variable):
    return [scalar_vector_product(alpha, row) for row in matrix_variable]
```


```python
matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
matrix_w = [[2, 5], [1, 1], [2, 2]]

print(scalar_matrix_product(3, matrix_x)) #Expected value: [[6, 6], [6, 6], [6, 6]]
print(scalar_matrix_product(2, matrix_y)) #Expected value: [[4, 10], [4, 2]]
print(scalar_matrix_product(4, matrix_z)) #Expected value: [[8, 16], [20, 12]]
print(scalar_matrix_product(3, matrix_w)) #Expected value: [[6, 15], [3, 3], [6, 6]]
```

    [[6, 6], [6, 6], [6, 6]]
    [[4, 10], [4, 2]]
    [[8, 16], [20, 12]]
    [[6, 15], [3, 3], [6, 6]]
    

# Problem 11.
두 개의 matrix가 입력 되었을 경우, 두 matrix의 곱셈 연산의 가능 여부를 True 또는 False로 반환  
A의 칼럼 수와 B의 로우 수가 같을 때만 곱셈이 가능


```python
def is_product_availability_matrix(matrix_a, matrix_b):
    return len(matrix_a[0]) == len(matrix_b)
```


```python
matrix_x= [[2, 5], [1, 1]]
matrix_y = [[1, 1, 2], [2, 1, 1]]
matrix_z = [[2, 4], [5, 3], [1, 3]]

print(is_product_availability_matrix(matrix_y, matrix_z)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_x)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_w)) # Expected value: False //matrix_w가없습니다
print(is_product_availability_matrix(matrix_x, matrix_x)) # Expected value: True
```

    True
    True
    False
    True
    

## Problem 12.
곱셈 연산이 가능한 두 개의 matrix의 곱셈을 실행하여 반환함


```python
def matrix_product(matrix_a, matrix_b):
    if is_product_availability_matrix(matrix_a, matrix_b) == False:
        return "Impossible"
    return [[sum(a*b for a,b in zip(row_a, column_b))] for column_b in zip(*matrix_b) for row_a in matrix_a]
```


```python
print(matrix_product(matrix_y, matrix_z)) # Expected value: [[9, 13], [10, 14]]
print(matrix_product(matrix_z, matrix_x)) # Expected value: [[8, 14], [13, 28], [5, 8]]
print(matrix_product(matrix_x, matrix_x)) # Expected value: [[9, 15], [3, 6]]
print(matrix_product(matrix_z, matrix_w)) # Expected value: False
```

    [[9], [10], [13], [14]]
    [[8], [13], [5], [14], [28], [8]]
    [[9], [3], [15], [6]]
    Impossible
    
