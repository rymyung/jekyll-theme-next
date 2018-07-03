---
title: 모델 성능 평가
date: 2018-07-03 17:27:00
categories:
- Algorithm
tags:
- General
- R
---
머신러닝을 통해 모델을 학습시킬 경우, 모델이 어느정도의 성능을 내느냐를 측정하는 것은 매우 중요하다. 또한, 모델은 이전에 본 적이 없는 데이터에도 일반화(Generalization)이 잘 되어 있어야 한다. 이번 포스트에서는 학습시킨 모델에 대한 성능 평가와 유효성 검증에 대해 알아보자.
* 모델의 성능 : 해결해야할 문제의 종류 - 분류(Classification) / 회귀(Regression) - 에 따른 평가 지표 선택
* 모델의 일반화(Generalization) : Bias-Variance / Overfitting 문제

# 모델 선정

모델을 선정하는 것은 머신러닝으로 어떠한 문제를 해결할 것인가에 달려있다.  
대표적인 머신러닝 문제 유형과 그 문제 유형을 해결하기 위한 모델들로는
* 분류
    * Naive Bayes
    * SVM
    * Decision Tree
    * Logistic Regression
    * ETC
* 회귀
    * Linear Regression
    * Genenralized Linear Regression
    * Decision Tree
    * ETC
* 비지도학습
    * K-Means
    * Association Rules
    * ETC  
등이 있다.

# 모델 성능 평가
모델 성능 평가는 모델의 성능을 검증하는 것이다. 이를 위해서는 모델의 목표와 사용된 모델링 기법 모두에 적절한 성능 평가 지표를 선택해야 한다.

## 회귀 모델 평가 지표
회귀 모델은 수치를 예측하는 모델으로, 실제 값과 예측 값과의 비교를 통해 모델의 성능을 평가한다.

### RMSE(Root Mean Square Error)

### 결정계수(\\(R^2 \\))

### MAE(Mean Absolute Error)

## 분류 모델 평가 지표
분류 모델은 2개 또는 그 이상의 범주를 예측하는 모델이다. 분류 모델의 성능을 측정하기 위해 사용하는 유용한 도구로 혼동 행렬(Confusion Matrix)가 있다.

### 혼동 행렬(Confusion Matrix)
혼동 행렬은 실제 알려진 데이터 범주에 대해 분류 모델의 예측을 정리한 표로, 각각의 예측 유형별로 실제 데이터가 얼마나 발생했는지 확인할 수 있다. 

혼동 행렬     | P'(Predict)  | N'(Predict) 
--------------|--------------|------------
**P**(Actual) |True Positive(**TP**) |False Negative(**FN**)   
**N**(Actual) |False Positive(**FP**)|True Negative(**TN**)   

혼동 행렬에서 True / False는 예측이 맞았는지/틀렸는지를 나타내고, Positive / Negative는 예측한 값이 1인지 / 0인지를 나타낸다.  
즉, 
* TP : 1이라고 예측하고 실제 값이 1인 경우
* FP : 1이라고 예측하고 실제 값이 0인 경우
* TN : 0이라고 예측하고 실제 값이 0인 경우
* FN : 0이라고 예측하고 실제 값이 1인 경우
이다.  

주로 관심이 가는 범주를 1로 간주하고 아닌 것은 0으로 간주한다.

연수입이 \$50K를 넘는지 안 넘는지를 분류하는 예제를 통해 혼동 행렬을 살펴보자. 이 데이터는 [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html)에서 가져왔다. 연령, 직업, 교육, 결혼상태, 성별, 인종 등을 변수로 연수입을 간략히 예측하는 분류 모델을 만들어보고 혼동 행렬을 만들었다. 혼동 행렬은 <code>caret</code> 패키지의 <code>confusionMatrix</code>를 사용하면 쉽게 구할 수 있다.


```R
# Load libraries
library(caret)
library(rpart)
library(e1071)

# Load data
adult <- read.csv("adult.csv")

# Split data
train.idx <- createDataPartition(adult$target, p = 0.7, list = F)
train <- adult[train.idx,]
test <- adult[-train.idx,]

# Train decision tree
dt.fit <- rpart(target ~ ., data = train, method = "class")

# Predict
train.pred <- predict(dt.fit, newdata = train, type = "class")
test.pred <- predict(dt.fit, newdata = test, type = "class")
```


```R
print(confusionMatrix(data = test.pred, reference = test$target, mode = "everything"))
```

    Confusion Matrix and Statistics
    
              Reference
    Prediction  <=50K  >50K
         <=50K   7114  1234
         >50K     302  1118
                                              
                   Accuracy : 0.8428          
                     95% CI : (0.8354, 0.8499)
        No Information Rate : 0.7592          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.5026          
     Mcnemar's Test P-Value : < 2.2e-16       
                                              
                Sensitivity : 0.9593          
                Specificity : 0.4753          
             Pos Pred Value : 0.8522          
             Neg Pred Value : 0.7873          
                  Precision : 0.8522          
                     Recall : 0.9593          
                         F1 : 0.9026          
                 Prevalence : 0.7592          
             Detection Rate : 0.7283          
       Detection Prevalence : 0.8546          
          Balanced Accuracy : 0.7173          
                                              
           'Positive' Class :  <=50K          
                                              
    

### 정확도(Accuracy)
정확도는 분류 모델의 성능을 측정하는데 사용하는 가장 일반적인 지표로, 정확히 분류된 항목의 숫자를 전체 항목의 숫자로 나눠서 구한다.  
$$\mathbf{Accuracy} = \frac{(\mathbf{TP} + \mathbf{TN})}{(\mathbf{TP} + \mathbf{TN} + \mathbf{FP} + \mathbf{FN})}$$

#### 클래스 불균형(Unbalanced Class) 문졔
암 진단과 같은 1과 0의 숫자가 매우 크게 차이가 나는 경우를 클래스 불균형 문제라고 한다. 이러한 문제에서는 대부분의 경우(99% 이상)이 0이기 때문에 정확도를 평가 지표로 하는 것은 좋은 방법이 아니다. 자세한 것은 [클래스 불균형]()에서 참고.

### 정밀도(Precision)

### 재현율(Recall)

### F1 Score

### 민감도(Sensitivity)

### 특이도(Specificity)

# 모델 유효성 검증
모델 유효성 검증은 모델이 train 데이터에서 잘 동작할 뿐만 아니라 새로운 데이터에 대해서도 잘 작동되어야 한다. 즉, 일반화(Generalization)이 잘 되어있어야 좋은 모델이 될 수 있다.

## Bias-Variance

## Overfitting

## Cross Validation

### Hold-out

### K-fold
