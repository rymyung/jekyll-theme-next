
다음 가이드는 구글의 R 스타일 가이드([Link](http://google.github.io/styleguide/Rguide.xml))을 기반으로 함

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
f
```


<pre class=language-r><code>function (x) 
x^2</code></pre>



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

