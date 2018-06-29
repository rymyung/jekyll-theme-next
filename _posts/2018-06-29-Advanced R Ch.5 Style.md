---
title: Advanced R - Chapter 5. Style
date: 2018-06-29 23:12:00
categories:
- Programming
tags:
- R
---

다음 가이드는 구글의 R 스타일 가이드([Link](http://google.github.io/styleguide/Rguide.xml))을 기반으로 함

# 표기법과 이름 짓기

## 파일 이름
파일 이름은 반드시 의미가 있어야 하며, .R로 끝나야함

* 좋은 경우 : fit-models.R / utility-functions.R  
* 나쁜 경우 : foo.r / stuff.r

파일들이 순서대로 실행될 필요가 있다면, 숫자로 된 접두사를 붙임
* 0-download.R
* 1-parse.R
* 2-explore.R

## 객체 이름
변수와 함수 이름은 반드시 소문자이어야하고, 이름 안에 있는 단어를 구분하려면 밑줄(\_)을 사용  
일반적으로 변수 이름은 명사이고, 함수 이름은 동사로 사용
* 좋은 경우 : day_one / day_1
* 나쁜 경우 : first_day_of_the_month / DayOne / dayone / djm1  
이미 존재하는 함수와 변수의 이름을 사용하는 것을 피하는게 좋음

# 문법

## 여백 주기
모든 삽입 연산자(=, +, -, <- 등) 주위에 여백(spaces)을 사용  
쉼표(,) 뒤에 항상 여백 삽입
* 좋은 경우 : <code>average <- mean(feet / 12 + inches, narm = TRUE)</code>  
* 나쁜 경우 : <code>average<-mean(feet/12+inches,na.rm=TRUE)</code>

이 규칙의 예외로 :, ::, 그리고 :::은 여백이 필요 없음
* 좋은 경우 : <code>base::get</code>
* 나쁜 경우 : <code>base :: get</code>

여백은 함수를 호출할 때는 넣지 않고, 왼쪽 괄호 앞에는 넣음
* 좋은 경우 : <code>if (debug) do(x)
plot(x, y)</code>
* 나쁜 경우 : <code>if(debug)do(x)
plot (x, y)</code>

등호나 할당(<\-)의 정렬을 개선하기 위해서라면, 추가 여백을 허용  
<code>list(total = a + b + c,
mean  = (a + b + c) / n)</code>

## 중괄호
중괄호가 시작되면 코드는 그 줄에서 끝나지 않고 새로운 줄이 이어짐  
중괄호를 닫으면 else가 뒤따르지 않는 한 그 줄에서 끝남
중괄호 안에서는 항상 들여쓰기 사용
* 좋은 경우 : <code>if (y < 0 && debug) {
    message("Y is negative")
}
if (y == 0) {
    log(x)
} else {
    y ^ x
}</code>
* 나쁜 경우 : <code>if (y < 0 && debug)
message("Y is negative")
if (y == 0) {
    log(x)
}
else {
    y ^ x
}</code>

매우 짧은 구문을 한 줄에 쓰는 것은 허용
<code>if (y < 0 && debug) message("Y is negative")</code>

## 줄 길이
코드를 가능한 한 줄에 80문자 이내로 한정  
출력 규격을 넘는 코드를 발견하면, 분리된 함수로 작업의 일부를 캡슐화하는 것을 고려

## 들여쓰기
코드를 들여쓰기(indentation)할 때는 여백 둘을 사용  
탭(tab)을 사용하거나 탭과 여백을 혼용하여 사용하지 말 것  
유일한 예외는 여러 줄에 걸쳐 함수를 정의하는 경우로, 이 경우에는 함수 정의가 시작되는 위치에 맞춰 두 번째 줄에 들여쓰기를 할 것
<pre><code>
long_function_name <- function(a = "a long arguement",
                               b = "another argument",
                               c = "another long argument") {
  \# 보통의 코드처럼 두 개의 여백으로 들여쓰기                               
}
<\/pre></code>

## 할당
할당에는 =를 사용하지 말고, <-를 사용
* 좋은 경우 : <code>x <- 5</code>
* 나쁜 경우 : <code>x = 5</code>

# 조직화

## 가이드라인 주석
코드에 주석(comment)를 달 것
주석의 각 줄은 주석 표시(#)와 여백 하나로 시작해야함
주석은 목적이 아니라 이유를 설명
파일을 분해해 쉽게 읽을 수 있는 묶음(chunks)으로 만들기 위해 -와 =로 주석이 달린 줄을 사용
<pre><code># Load data -----------------------<\/pre></code>
<code># Plot data -----------------------</code>
