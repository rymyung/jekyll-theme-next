---
title: Advanced R - Chapter 6. Functions
date: 2018-07-02 12:27:00
categories:
- Programming
tags:
- R
---

이 포스트는 책 *Advanced R*의 *Chapter 6*을 요약한 포스트입니다. [Link](http://adv-r.had.co.nz/)

# Intro
함수는 R을 구성하는 근본으로, 함수가 동작하는 방식을 알아야 R의 고급 기법들에 익숙해질 수 있다.  
이 장의 초점은 기존의 함수에 대한 비공식적인 지식에서 벗어나 함수란 무엇이며, 이 함수는 어떻게 동작하는지를 정확하게 이해하도록 하는 것이다.  
R을 이해하는데 가장 중요한 점은 함수 그 자체가 객체라는 거시다. 어떤 다른 유형의 객체를 다루는 것과 정확하게 동일한 방법으로 함수를 다룰 수 있다.  

## 사전 준비
*pryr* 패키지는 벡터를 수정할 때 그곳에서 어떤 일이 발생하는지 탐색하기 위해 사용한다.


```R
library(pryr)
```

# 함수 구성 요소
모든 R 함수는 세 부분으로 구성되어 있다.
* body(), 함수 안에 쓰인 코드
* formals(), 함수 호출을 제어하는 인자 목록
* environment(), 함수의 변수에 대한 위치 지도


```R
f <- function(x) x^2
print(f)
```

    function(x) x^2
    


```R
formals(f)
```


    $x
    
    



```R
body(f)
```


    x^2



```R
environment(f)
```


    <environment: R_GlobalEnv>


## 원시 함수
함수가 세 가지 요소를 가진다는 규칙에는 한 가지 예외가 있다. <code>sum()</code>과 같은 원시 함수(primitive functions)는 <code>.Primitive()</code>로 직접 C코드를 호출하는데, 이와 같은 함수에는 R 코드가 없다. 따라서 함수의 formals(), body(), 그리고 environment()가 모두 <code>NULL</code>이다.


```R
print(sum)
```

    function (..., na.rm = FALSE)  .Primitive("sum")
    


```R
formals(sum)
```


    NULL



```R
body(sum)
```


    NULL



```R
environment(sum)
```


    NULL


# 렉시칼 스코핑(Lexical Scoping)
스코핑은 R이 심볼 값을 찾는 방법을 지배하는 규칙들의 집합으로, 스코핑에 대한 이해는 다음을 수행하는데 도움이 된다.
* 함수들을 합성하여 도구를 구축하기(9장)
* 일반적 평가 규칙을 억제하고 비표준적 평가를 실행하기(12장)  

렉시칼 스코핑은 함수가 호출됐을 때가 아니라 생성되었을 때 **중첩되는**(nested) 방식에 따라 심볼 값을 찾는다. 렉시칼 스코핑을 사용하면 변수의 값을 어디에서 찾아야 할지 알아내기 위해 함수를 호출하는 방법을 알아야 할 필요가 없고, 단지 함수의 정의만 살펴보면 된다.

스코핑 방법의 종류
* 렉시칼 스코핑 : 언어 수준에서 자동적으로 구현
* 다이내믹 스코핑 : 인터랙티브한 분석 중 타이핑을 효율적으로 하기 위한 함수 선택에 사용(12.3절)
으로 나눌 수 있다.

렉시칼 스코핑에는 4가지 기본적인 원칙이 있다.
* 이름 마스킹(name masking)
* 함수와 변수(functions vs. variables)
* 새롭게 시작(a fresh start)
* 동적 탐색(dynamic lookup)

## 이름 마스킹(name masking)
함수 내에 정의된 이름을 탐색하여 심볼 값을 찾는다.


```R
f <- function() {
    x <- 1
    y <- 2
    c(x, y)
}
f()
rm(f) # f 객체 제거
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
</ol>



함수 내에 이름이 정의되지 않았다면, 한 수준 위를 탐색한다.


```R
x <- 2
g <- function() {
    y <- 1
    c(x, y)
}
g()
rm(x, g) # x, g 객체 제거
```


<ol class=list-inline>
	<li>2</li>
	<li>1</li>
</ol>



어떤 함수가 다른 함수 안에 정의되어도 이와 동일한 규칙이 적용된다. 현재 함수의 내부를 탐색하고 난 후 함수가 정의된 곳을 찾고, 이를 반복해 전역 환경에까지 가는 모든 방법을 거치고 로딩된 다른 패키지들을 찾는다.


```R
x <- 1
h <- function() {
    y <- 2
    i <- function() {
        z <- 3
        c(x, y, z)
    }
    i()
}
h()
rm(x, h) # x, h 객체 제거
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>3</li>
</ol>



k() 함수가 호출된 이후에 y의 값을 알 수 있는 이유는, k가 그 정의된 환경을 유지하고 있고, 그 환경은 y의 값을 포함하기 때문이다.


```R
j <- function(x) {
    y <- 2
    function() {
        c(x, y)
    }
}
k <- j(1)
k()
#rm(j, k) # j, k 객체 제거
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
</ol>



## 함수와 변수
함수 검색은 변수를 찾는 방법과 정확히 동일한 방법으로 동작한다.


```R
l <- function(x) x + 1
m <- function() {
    l <- function(x) x * 2
    l(10)
}
m()
rm(l, m) # l, m 객체 제거
```


20


함수의 경우, 규칙에 대한 한 가지 작은 **변형**이 있다. 원하는 것이 함수인 게 명백할 때 이름을 사용하면, 검색하는 동안 함수가 아닌 객체들을 무시한다.


```R
n <- function(x) x / 2
o <- function() {
    n <- 10
    n(n)
}
o()
rm(n, o) # n, o 객체 제거
```


5


그러나 함수와 다른 객체에 대해 동일한 이름을 사용하면 코드가 혼란스러워지므로 일반적으로 이런 경우를 피하는 것이 좋다.

## 새롭게 시작
함수가 매번 호출될 때마다 호스트를 실행하기 위해 새로운 환경을 생성한다. 즉, 각 함수의 호출은 완벽하게 독립적이다.


```R
j <- function() {
    if (!exists("a")) {
        a <- 1
    } else {
        a <- a + 1
    }
    print(a)
}
j()
rm(j) # j 객체 제거
```

    [1] 1
    

## 동적 탐색
R은 함수가 생성될 때가 아니라 실행될 때 값을 탐색한다. 즉, 함수의 출력이 환경의 외부에 있는 객체에 따라 달라질 수 있다.


```R
f <- function() x
x <- 15
f()
```


15



```R
x <- 20
f()
```


20


코드를 작성할 때 철자에 오류가 있다면 함수를 생성할 때 오류가 나타나지 않으므로  
어떤 변수가 전역 환경에서 정의되는지에 따라 그 함수를 실행할 때에도 오류가 나지 않는다.

이런 문제를 발견하는 한 가지 방법은 **codetools**의 <code>findGlobals()</code> 함수를 사용하는 것이다.  
이 함수는 특정 함수의 모든 외부 **의존성**(dependency)을 나열한다.


```R
f <- function() x + 1
codetools::findGlobals(f, F)
```


<dl>
	<dt>$functions</dt>
		<dd>'+'</dd>
	<dt>$variables</dt>
		<dd>'x'</dd>
</dl>




```R
f2 <- function(x) x * 2
codetools::findGlobals(f2, F)
```


<dl>
	<dt>$functions</dt>
		<dd>'*'</dd>
	<dt>$variables</dt>
		<dd></dd>
</dl>



다른 방법은 <code>emptyenv()</code>로 함수의 환경을 아무 것도 담고 있지 않은 환경으로 직접 변경하는 것이다.


```R
codetools::findGlobals(f2, F)
```


<dl>
	<dt>$functions</dt>
		<dd>'+'</dd>
	<dt>$variables</dt>
		<dd></dd>
</dl>




```R
environment(f) <- emptyenv()
f()
```


    Error in x + 1: 함수 "+"를 찾을 수 없습니다
    Traceback:
    

    1. f()


R은 +연산자를 포함한 모든 것을 찾을 때 렉시칼 스코핑에 의존하기 때문에 위의 <code>f</code> 함수는 동작하지 않는다. 베이스 R 또는 다른 패키지들에서 정의된 함수에 항상 의존해야하기 때문에 완벽한 자기 포함적 함수를 만들 수 없다.

# 모든 연산은 함수 호출
R에서의 계산을 이해하는 데는 다음 두 가지 슬로건이 도움이 된다.
>* "존재하는 모든 것은 객체다."
* "발생하는 모든 것은 함수 호출이다."

# 함수 인자
함수의 형식 인자(formal arguments)와 실질 인자(actual arguments)를 구분하는 것이 유용하다.
* 형식 인자 : 하나의 함수 속성
* 실질 인자 또는 호출 인자(calling arguments) : 함수를 호출할 때마다 변함

## 함수 호출
함수를 호출할 때 위치, 전체 이름, 또는 부분 이름으로 인자를 특정할 수 있다. 인자는 처음에는 정확한 이름(완전 매칭 : perfect matching)으로, 그 다음에는 접두어 매칭(prefix matching)으로, 그리고 마지막에는 위치로 매칭된다. **이름 있는 인자**(named arguments)는 **이름 없는 인자**(unnamed arguments)의 뒤에 위치해야 한다.


```R
# 위치 매칭
f <- function(abcdef, bcde1, bcde2) {
    list(a = abcdef, b1 = bcde1, b2 = bcde2)
}
str(f(1,2,3))
```

    List of 3
     $ a : num 1
     $ b1: num 2
     $ b2: num 3
    


```R
# 전체 이름 매칭
str(f(2,3,bcde1 = 1))
```

    List of 3
     $ a : num 2
     $ b1: num 1
     $ b2: num 3
    


```R
# 부분 이름 매칭
str(f(2,3,a=1))
```

    List of 3
     $ a : num 1
     $ b1: num 2
     $ b2: num 3
    


```R
# 오류 발생
str(f(1,3,b=1))
```


    Error in f(1, 3, b = 1): argument 3 matches multiple formal arguments
    Traceback:
    

    1. str(f(1, 3, b = 1))


## 주어진 인자 목록에서 함수 호출
함수 인자 목록이 주어져 있는 경우, <code>do.call()</code>를 사용하여 인자를 함수에 보낼 수 있다.


```R
args <- list(1:10, na.rm = TRUE)
do.call(mean, args)
# 위와 동일
mean(1:10, na.rm = TRUE)
```


5.5



5.5


## 기본 값과 결측 인자
R의 함수 인자는 기본 값을 가질 수 있다.


```R
f <- function(a = 1, b = 2) {
    c(a,b)
}
f()
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
</ol>



R에서 인자는 느슨하게 평가되기 때문에, 기본값이 다른 인자로 정의될 수 있다.


```R
g <- function(a = 1, b = a * 2) {
    c(a, b)
}
g()
g(10)
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
</ol>




<ol class=list-inline>
	<li>10</li>
	<li>20</li>
</ol>



기본 인자는 함수 내에 생성된 변수로도 정의될 수 있다. 이 것은 베이스 R 함수에서 자주 사용되지만, 완벽하게 소스 코드를 읽지 않으면 기본값이 무엇인지 알 수 없기 때문에 좋지 않은 방법이다.


```R
h <- function(a = 1, b = d) {
    d <- (a + 1) ^ 2
    c(a, b)
}
h()
h(10)
```


<ol class=list-inline>
	<li>1</li>
	<li>4</li>
</ol>




<ol class=list-inline>
	<li>10</li>
	<li>121</li>
</ol>



<code>missing()</code> 함수를 사용하면 인자가 제공되었는지 여부를 판단할 수 있다.


```R
i <- function(a, b) {
    c(missing(a), missing(b))
}
i()
i(a = 1)
i(b = 2)
i(1, 2)
```


<ol class=list-inline>
	<li>TRUE</li>
	<li>TRUE</li>
</ol>




<ol class=list-inline>
	<li>FALSE</li>
	<li>TRUE</li>
</ol>




<ol class=list-inline>
	<li>TRUE</li>
	<li>FALSE</li>
</ol>




<ol class=list-inline>
	<li>FALSE</li>
	<li>FALSE</li>
</ol>



## 느슨한 평가
R 함수 인자는 오로지 실제로 사용될 때 평가된다.


```R
f <- function(x) {
    10
}
f(stop("This is an error!"))
```


10


인자가 평가되었는지 확인하려면 force()를 사용해야 한다.


```R
f <- function(x) {
    force(x)
    10
}
f(stop("This is an error!"))
```


    Error in force(x): This is an error!
    Traceback:
    

    1. f(stop("This is an error!"))

    2. force(x)   # at line 2 of file <text>

    3. stop("This is an error!")



```R
add <- function(x) {
    function(y) x + y
}
adders <- lapply(1:10, add)
```


```R
adders[[9]](10)
```


19



```R
add <- function(x) {
    force(x)
    function(y) x + y
}
adders2 <- lapply(1:10, add)
```


```R
adders2[[9]](10)
```


19


## ...
...라는 특수한 인자는 매칭되지 않은 다른 어떤 인자와도 매칭될 수 있기 때문에 쉽게 다른 함수에 적용할 수 있다. 다른 함수를 호출하기 위해 인자를 선택하기를 원하지만, 그 가능한 이름을 사전에 정의하기 윈치 않을 때 유용하다. ...는 종종 개별적인 메소드를 보다 유연하게 해주는 **S3 제너릭 함수**(S3 generic functinos)와 결합하여 사용한다.

기본 <code>plot()</code>함수는 ...인자를 비교적 복잡하게 사용한다. <code>plot()</code>는 x, y, 그리고 ...를 인자로 가지는 **제너릭 메소드**(generic method)이다. ...이 <code>par()</code> 도움말에 나열돼 있는 **다른 그래피 파라미터**를 허용한다는 것을 알 수 있다. 이에 따라 다음과 같은 코드를 작성할 수 있다.


```R
plot(1:5, col = "red")
plot(1:5, cex = 5, pch = 20)
```

이 것은 ...의 장점과 단점을 동시에 보여 준다. 즉, 이와 같은 방법은 <code>plot()</code>을 매우 유연하게 해주지만, 사용법을 이해하려면 문서를 주의 깊게 살펴봐야 한다.

...의 사용은 어떤 잘못 표기된 인자도 오류를 나타내지 않기 때문에 ... 뒤의 모든 인자는 완전한 이름이 있어야 한다. 이 것은 인지하지 못한 오타를 만들기 쉽게 한다.


```R
sum(1, 2, NA, na.mr = TRUE) # na.rm 대신 na.mr을 사용
```


&lt;NA&gt;


# 특수한 호출
R은 함수를 호출하는 특수한 유형의 추가 구문 두 가지를 지원한다.
* 삽입 함수(infix functions)
* 대체 함수(replacement functions)

## 삽입 함수
대부분의 R 함수는 접두 연산지(prefix operators)이므로 함수의 이름이 인자 앞에 온다. 그러나 + 또는 -와 같이 함수명이 그 인자 사이로 오는 삽입 함수를 생성할 수도 있다. 모든 사용자 생성 삽입 함수는 %로 시작하고 끝나야 한다. R에는 다음과 같이 미리 정의된 삽입 함수가 있다.: <code>%%, %*%, %/%, %in%, %o%, %x%</code>


```R
`%+%` <- function(a, b) paste0(a, b)
"new" %+% " string"
`%+%`("new", " string")
```


'new string'



'new string'


삽입 함수의 이름은 정규 R 함수에 비해 유연해서 어떤 문자 시퀀스도 포함할 수 있다.(% 제외)


```R
`% %` <- function(a, b) paste(a, b)
`%'%` <- function(a, b) paste(a, b)
"a" % % "b"
"a" %'% "b"
```


'a b'



'a b'


## 대체 함수
대체 함수는 함수가 그 함수의 인자들을 수정하는 것처럼 동작하고, <code>xxx <- </code>라는 특수한 이름을 가진다. 대체 함수는 두 개의 인자(x와 value)를 가지며, 수정된 객체를 반환한다.


```R
`second<-` <- function(x, value) {
    x[2] <- value
    x
}
x <- 1:10
second(x) <- 5L # x 벡터의 두 번째 요소를 수정
x
```


<ol class=list-inline>
	<li>1</li>
	<li>5</li>
	<li>3</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
	<li>7</li>
	<li>8</li>
	<li>9</li>
	<li>10</li>
</ol>



대체 함수가 그 함수의 인자를 수정하는 것처럼 동작한다고 말한 이유는 실제로 수정된 사본을 생성하기 때문이다.


```R
x <- 1:10
address(x)
```


'0x363ce300'



```R
second(x) <- 6L
address(x)
```


'0xbb754c8'


추가 인자를 삽입하려면 그 인자가 x와 value 사이에 있어야 한다.


```R
`modify<-` <- function(x, position, value) {
    x[position] <- value
    x
}
x <- 1:10
modify(x, 1) <- 10
print(x)
```

     [1] 10  2  3  4  5  6  7  8  9 10
    

<code>modify(x, 1) <- 10</code>을 호출할 때 R은 이면에서 다음과 같이 동작한다.<code>x <- `modify<-`(x, 1, 10)</code>

# 반환값
함수 내에서 평가된 마지막 표현식은 함수 기동의 결과인 **반환값**(return value)이 된다.


```R
f <- function(x) {
    if (x < 10) {
        0
    } else {
        10
    }
}
f(5)
```


0


일반적으로 오류나 단순한 함수의 경우처럼 일찍 반환되는 경우 명백하게 <code>return()</code>을 사용하는 것이 좋은 코딩 스타일이다.


```R
f <- function(x, y) {
    if (!x) return(y)
    # 여기에 복잡한 프로세스를 삽입
}
```

함수는 오로지 하나의 객체만을 반환할 수 있으나, 여러 객체를 담고 있는 리스트도 반환할 수 있다.

## 나가기
함수는 값을 반환할 수 있고 <code>on.exit()</code>를 사용하여 끝날 때 시작되는 다른 트리거를 설정할 수도 있다. 이 것은 함수가 종료될 때 전역 상태에 대한 변경사항을 확실히 복원하는데 사용한다. <code>on.exit()</code> 안의 코드는 함수가 종료된 방법, 즉 명시적 (초기) 반환, 오류, 또는 단순히 함수 본문의 끝까지 도달했는지의 여부에 관계 없이 실행된다.


```R
in_dir <- function(dir, code) {
    old <- setwd(dir)
    on.exit(setwd(old))
    
    force(code)
}
getwd()
```


'C:/Users/Ro_Laptop/Dropbox/Public/공부/github/advancedR'



```R
in_dir("~", getwd())
```


'C:/Users/Ro_Laptop/Documents'

