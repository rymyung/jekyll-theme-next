---
title: 선형 회귀 분석(Linear Regression)
date: 2018-07-12 18:02:00
categories:
- Algorithm
tags:
- Linear Regression
- Statistics
- R
---
선형 회귀란 종속 변수와 한 개 이상의 독립(설명) 변수와의 선형 상관 관계를 모델링하는 기법이다.
$$ y = \beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 + \cdots +  \beta_n \cdot x_n + \varepsilon$$
<center> 또는 </center>
$$H(X) = W^TX$$
선형 회귀는 회귀계수 $\beta_n$, $W$을 추정하는 선형 관계를 가정한 모수적 방법으로, 선형회귀의 목적은 크게 2가지가 있다.
* Predictive Model : 여러 변수들을 가지고 하나의 변수에 대해 예측하려고 할 때(예측력이 중요)
* Explanatory Model : 둘 혹은 여럿 간의 관계를 설명하거나 이해하고자 할 때(종속 변수와 독립 변수 사이의 인과관계가 중요)

R의 <code>mlbench</code>패키지에 내장되어 있는 보스턴 지역 주택 가격 데이터로 회귀 분석을 해보자.


```R
# Load library
library(mlbench)

# Load data
data(BostonHousing)
print(head(BostonHousing))
```

         crim zn indus chas   nox    rm  age    dis rad tax ptratio      b lstat
    1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98
    2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14
    3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03
    4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94
    5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33
    6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21
      medv
    1 24.0
    2 21.6
    3 34.7
    4 33.4
    5 36.2
    6 28.7
    

BostonHusing 데이터 안에 있는 변수들을 사용해 마지막 변수인 medv를 예측하는 선형 회귀를 학습시키는 방법은 다음과 같다.


```R
# Split data
train.id <- sample(1:nrow(BostonHousing), 0.7*nrow(BostonHousing))
train    <- BostonHousing[train.id,]
test     <- BostonHousing[-train.id,]

# Train linear model
lm.fit   <- lm(medv ~ ., data = train)
summary(lm.fit)

# Predict
lm.pred <- predict(lm.fit, newdata = test)
```


    
    Call:
    lm(formula = medv ~ ., data = train)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -11.0704  -2.7449  -0.8038   1.6047  24.2637 
    
    Coefficients:
                  Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  3.574e+01  6.341e+00   5.636 3.66e-08 ***
    crim        -1.522e-01  4.169e-02  -3.651 0.000303 ***
    zn           4.245e-02  1.703e-02   2.493 0.013131 *  
    indus        5.698e-02  7.968e-02   0.715 0.475033    
    chas1        3.842e+00  1.131e+00   3.396 0.000766 ***
    nox         -1.527e+01  4.723e+00  -3.233 0.001345 ** 
    rm           3.610e+00  5.160e-01   6.995 1.42e-11 ***
    age          1.869e-04  1.558e-02   0.012 0.990440    
    dis         -1.381e+00  2.389e-01  -5.782 1.68e-08 ***
    rad          3.833e-01  8.350e-02   4.591 6.23e-06 ***
    tax         -1.317e-02  4.843e-03  -2.720 0.006859 ** 
    ptratio     -9.373e-01  1.626e-01  -5.765 1.83e-08 ***
    b            9.682e-03  3.332e-03   2.906 0.003900 ** 
    lstat       -5.819e-01  6.092e-02  -9.552  < 2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Residual standard error: 4.881 on 340 degrees of freedom
    Multiple R-squared:  0.7276,	Adjusted R-squared:  0.7172 
    F-statistic: 69.88 on 13 and 340 DF,  p-value: < 2.2e-16
    


각 변수의 회귀 계수에 따라 회귀 식을 구성하면,
$$ \text{medv} = (\text{ } -0.098797 * \text{crim }) + (\text{ } 0.034875 * \text{zn }) + \cdots + (\text{ } -0.618872 * \text{lstat })$$
이 된다. 간단히 해석해보면, 
* crim 변수가 한 단위 증가할 때마다 medv는 0.098797만큼 감소한다.
* zn 변수가 한 단위 증가할 때마다 medv는 0.0348757만큼 증가한다.
* $\vdots$
* lstat 변수가 한 단위 증가할 때마다 medv는 0.618872만큼 감소한다.

선형 회귀 모델을 만들었으면, 이 모델이 적절한 것인지에 대한 검증이 필요하다. 회귀 모형은 기본적으로 여러 가정들을 기반으로 한 통계 모형이기 때문에 이러한 가정을 무시하고 모델을 만들었을 경우, 부정확한 모델이 될 수 있다.

# 오차항 $ \varepsilon$ (Random error term)
오차는 회귀식을 중심으로 무작위하게 흩어져 있는 변동으로, 다음과 같은 가정을 바탕으로 한다.
* $\varepsilon_i$는 정규분포의 형태를 이룬다.
* $\varepsilon_i$의 기대값은 0이다, $E(\varepsilon_i) = 0$. 이 가정은 실제 값이 회귀선 상에 있는 점을 중심으로 분포되어 있다는 뜻이다.
* $\varepsilon_i$의 분산은 모든 $x$에 대해 동일하다, $\sigma^2(\varepsilon_i) = \sigma^2$
* $\varepsilon_i$들은 서로 독립적이다.

![선형회귀도해](https://www.dropbox.com/s/yxan4ive93j4yyn/error%20term.jpg?raw=1)
<center>[이미지 출처](http://reliawiki.org/index.php/Simple_Linear_Regression_Analysis)</center>

오차항 $\varepsilon_i$가 확률변수이므로, $y_i$도 확률변수가 된다. 또한 $E(\varepsilon_i) = 0$이므로
$$E(y_i) = E(\beta_0 + \beta_1x_1 + \cdots + \beta_nx_n + \varepsilon_i) = \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n$$가 된다. 위 식을 모든 x와 y에 대하여 나타내면,
$$E(y) = \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n$$가 되는데 이 것이 바로 회귀 함수가 된다.

## 잔차 $e_i$ (residual)
실제값($y_i$)과 예측값($\hat{y_i}$)의 차이를 잔차라고 한다.
$$e_i = y_i - \hat{y_i}$$
잔차 $e_i$와 오차항$\varepsilon_i$가 어떻게 다른가를 구별하는 것은 중요한데, 모형을 다시 한 번 정리해보면 다음과 같다.
$$y_i = \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n + \varepsilon_i$$
$$y_i = \hat{\beta_0} + \hat{\beta_1}x_1 + \cdots + \hat{\beta_n}x_n + e_i$$
즉,
$$\varepsilon_i = y_i - (\beta_0 + \beta_1x_1 + \cdots + \beta_nx_n)$$
$$e_i = y_i - (\hat{\beta_0} + \hat{\beta_1}x_1 + \cdots + \hat{\beta_n}x_n)$$이다.
위에서 알 수 있듯이 오차항 $\varepsilon_i$는 실제값 $y_i$와 **모집단** 회귀식과의 차이를 말하고, 잔차 $e_i$는 실제값 $y_i$와 **추정** 회귀식과의 편차를 말한다.  
오차항 $\varepsilon_i$는 실제로 알 수 없기 때문에 잔차 $e_i$에 의해 추정된다.

# 선형 회귀에서의 가정
선형 회귀에서는 기본적으로 다음과 같은 4가지 가정을 바탕으로 한다.
* 정규성(Normality) : 평균이 0인 정규분포를 따름, $\varepsilon \text{ ~ } N(0, \sigma^2)$
* 독립성(Independence) : 오차항은 서로 독립
* 선형성(Linearity) : 종속 변수와 독립 변수가 선형 관계
* 등분산성(Homoscedasticity) : 오차항은 모든 $x$에 대해 동일한 분산을 가짐


```R
par(mfrow = c(2,2))
plot(lm.fit)
```


![png](output_10_0.png)


위의 4가지 그래프를 통해 선형 회귀의 기본 가정들에 대한 검증을 할 수 있다.
* Residual vs Fitted : 잔차와 예측값을 산점도로, 독립성과 등분산성 확인 가능
* Normal Q-Q :정규성을 확인 가능
* Scale-Location : 표주화된 잔차와 예측값의 산점도로, REsidual vs Fitted plot과 유사
* Residuals vs Leverage : 표준화된 잔차와 지레값(leverage)의 산점도로, x값과 y값의 이상치 확인 가능

## 정규성
선형 회귀에서는 오차항 $\varepsilon_i$들이 정규분포를 이루고 있어야 한다. 그러므로 잔차 또한 평균이 0인 정규분포를 이루고 있어야 한다. normal Q-Q plot은 표준화된 잔차의 probability plot으로, 만약 잔차가 정규성 가정을 만족한다면 이 그래프의 점들은 45도 각도의 직선 상에 위치해야 한다. R의 <code>car</code> 패키지의 <code>qqPlot()</code>함수는 정규성 가정을 확인하는데 더 정확한 방법을 제공한다.


```R
# Load library
library(car)

# Ignore warning
options(warn = -1)

# Plot
qqPlot(lm.fit)
```


![png](output_13_0.png)


그러나 정규분포가 아니더라도 정규분포에서 크게 벗어나지만 않으면 문제가 생기지 않으므로, 잔차 $e_i$들을 개략적으로 검토하여 정규성을 판단해도 무방하다.


```R
# Load libraries
library(ggplot2)
library(dplyr)

# Control plot size
options(repr.plot.width=4, repr.plot.height=3)

# Calculate residuals
e <- rstandard(lm.fit)

# Plot
data.frame(e = e) %>%
ggplot() + geom_density(aes(e, fill = 'red', col = 'red'), alpha = 0.2) + theme_minimal() + theme(legend.position = "none")
```




![png](output_15_1.png)


## 독립성
선형 회귀에서는 $\varepsilon_i$들이 서로 독립적이어야 한다. 독립성을 만족하면, 다음 plot에서 어떤 관계가 보이지 않는다. 


```R
plot(lm.fit, which = c(1))
```


![png](output_17_0.png)


 데이터가 수집될 때, 시간 간격을 두고 계속적으로 수집될 경우 $\varepsilon_i$들이 서로 독립적이지 않고 자기상관성(autocorrelation)을 가질 수가 있다. <code>car</code> 패키지의 durbin-Watson 검정으로 독립성을 판단할 수 있다. 이 함수의 결과가 유의하게 나온다면 자기상관성이 있다고 판단할 수 있다.


```R
durbinWatsonTest(lm.fit)
```


     lag Autocorrelation D-W Statistic p-value
       1      0.06704547      1.860615   0.194
     Alternative hypothesis: rho != 0


## 선형성
선형 회귀에서는 오차항들이 서로 독립이어야한다. <code>car</code> 패키지의 <code>crPlots()</code> 함수를 사용해 선형성을 확인할 수 있다.


```R
crPlots(lm.fit)
```


![png](output_21_0.png)



![png](output_21_1.png)


위의 plot들에서 비선형성이 관찰된다면 변수 변환을 하거나 비선형 회귀 모형을 사용해야 한다.

## 등분산성
선형 회귀에서는 $\varepsilon_i$들의 분산이 모두 같아야 한다. 등분산성을 만족하면, 수평선 주위에 Random Band 형태로 어떤 관계가 보이지 않는다. 만약 등분산성 가정이 위배된다면 변수 변환을 통해서 독립 변수들을 변환시켜서 회귀 모델에 적용해야 한다.


```R
plot(lm.fit, which = c(3))
```


![png](output_24_0.png)


<code>car</code> 패키지의 <code>ncvTest()</code> 함수는 예측값에 따라 오차의 분산이 변하는지를 검정해주는 함수이다. 이 함수의 결과가 유의하게 나온다면 등분산성 가정이 위배된다고 할 수 있다.


```R
ncvTest(lm.fit)
```


    Non-constant Variance Score Test 
    Variance formula: ~ fitted.values 
    Chisquare = 7.356721    Df = 1     p = 0.006681252 


## 회귀 계수 $\beta$의 추정 ($\hat{\beta}$)
최소제곱법(Ordinary Least Squares, OLS)은 가장 기본적인 회귀 계수를 추정하는 방법으로 실제 관측값($y_i$)과 예측값($\hat{y_i}$)과의 차이를 최소로 하는 $\beta$를 추정하는 방법이다. 잔차에는 $\pm$부호가 있으므로 각 잔차를 제곱하여 더한 합을 최소로 하는 $\beta$를 추정한다.
$$ \text{잔차제곱합(Residual Sum of Squares, RSS) : } \sum_{i=1} e_i^2 = \sum_{i=1} (y_i - \beta_ix_i)^2$$
최소제곱법 이외에 일반화 최소제곱법, 최우추정법, 도구변수 추정법 등 회귀 계수를 추정하는 여러 방법이 있다. 어떤 방법을 쓰는 것이 좋은가는 결국 모집단에 대해 어떤 가정을 할 수 있는가에 달려있다.

비용 함수(cost function) 혹은 손실 함수(loss function)를 최소로하는 가중치 벡터($W$)를 구하는데 **gradient descent algorithm**를 사용한다. 
$$cost(W, b) = \frac{1}{m}\sum_{i=1}^m (H(x_i) - y_i)^2$$


```R
# Load library
library(ggplot2)

# Generate data
x <- seq(-10, 10, 1)
y <- 0.5*x + rnorm(21)
lm.df <- data.frame(x = x, y = y)

# Plot
ggplot(lm.df, aes(x, y)) + geom_point() + theme_minimal() + geom_abline(intercept = 0, slope = 0.5, col = "blue") +
geom_segment(aes(x, 0.5*x, yend = y, xend = x)) 
```




![png](output_29_1.png)


다음의 예시에서, x와 y는 선형 관계로 표현될 수 있고 그 가중치는 0.5이다. 이제 여러 개의 가중치들로 비용 함수를 계산해보자.


```R
# Calculate cost with different w
cost_list <- c()
for (w in seq(-3, 4, 0.1)) {
    y_hat <- w * x
    cost <- mean((y_hat - y)^2)
    cost_list <- c(cost_list, cost)
}
cost.df <- data.frame(w = seq(-3, 4, 0.1), cost = cost_list)

# Plot
ggplot(cost.df, aes(w, cost)) + geom_point() + 
geom_line(y = 0, col = 'red') + theme_minimal()
```




![png](output_31_1.png)


-3부터 4까지 0.1 씩 증가시키면서 가중치를 주었을 때 비용 함수는 위의 그래프와 같다. 이 그래프에서 cost가 가장 작은 w는 0.5이다.

## Gradient Descent

1. Start with initial point(anywhere)
2. W를 조금씩 바꾸면서 cost를 감소
3. Local minimum에 converge될 때까지 반복

계산의 편의를 위해 cost function을 약간 변형하여 사용한다.
$$cost(W, b) = \frac{1}{m}\sum_{i=1}^m (H(x_i) - y_i)^2$$
$$\Downarrow$$
$$cost(W, b) = \frac{1}{2m}\sum_{i=1}^m (H(x_i) - y_i)^2$$

처음에는 아무 가중치 벡터를 설정하여 
$$W := W - \alpha \frac{\partial}{\partial W}cost(W), \alpha \text{ : learning rate}$$
$$W := W - \alpha \frac{\partial}{\partial W}\frac{1}{2m}\sum_{i=1}^m (Wx_i - y_i)^2$$
$$\Downarrow$$
$$W := W - \alpha \frac{\partial}{\partial W}\frac{1}{2m}\sum_{i=1}^m 2(Wx_i - y_i)x_i$$
$$\Downarrow$$
$$W := W - \alpha \frac{\partial}{\partial W}\frac{1}{m}\sum_{i=1}^m (Wx_i - y_i)x_i$$

# 적합도 검정

# 선형 회귀에서의 고려사항들

## 범주형 변수

## 비선형성

## 다중공선성
