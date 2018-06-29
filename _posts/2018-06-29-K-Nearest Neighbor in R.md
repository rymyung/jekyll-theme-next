---
title: K-Nearest Neighbor
date: 2018-06-29 23:12:00
categories:
- Algorithm
    - Classification
tags:
- knn
- R
---

# K-NN
기존의 데이터 중 가장 유사한 k개의 데이터를 이용해서 새로운 데이터를 예측하는 방법

* Continuous: k개 데이터의 평균
* Categorical : k개 데이터 중 가장 많이 나온 클래스
![knn](https://www.dropbox.com/s/6scqv9dt9voikhw/knn1.jpg?raw=1)
K가 3인 분류 문제의 경우, 가장 가까운 3개의 데이터의 클래스를 다수결에 따라 새로운 데이터의 클래스로 지정

## 게으른 학습
모델을 생성하지 않고 분류가 필요할 때마다 모든 데이터에 대해 유사도를 측정해야 하기 때문에 속도가 매우 느림

## 유사도(Similarity)
각 데이터의 유사도를 측정하는 방법들  
범주형 변수를 포함하는 경우에는 가능한 k-NN을 사용하지 않거나 Hamming Distance 사용

### 유클리디언 거리
연속 변수에서 가장 많이 사용되는 척도
$$d(x,y) = \sqrt[2]{\sum_{i=1}^n (x_i-y_i)^2}$$

### 맨하튼 거리
![Manhattan](https://www.dropbox.com/s/h0omd4awsuqo4pq/knn2.jpg?raw=1)
$$d(x,y) = \sum_{i=1}^n [x_i-y_i]$$

### 민코프스키 거리
$$d(x,y) = \sqrt[r]{\sum_{i=1}^n [x_i-y_i]^r} \begin{cases} \text{Manhattan}, & \text{if r} = 1 \\ \text{Euclidean}, & \text{if r} = 2 \end{cases}$$

### 단순 매칭 계수(Simple Matching Coefficient, SMC)
변수 x와 y가 이진 속성만을 가진 경우 유사도를 측정하는 방법
* f00 : x와 y가 모두 0인 수
* f01 : x가 0이고 y가 1인 수
* f10 : x가 1이고 y가 0인 수
* f11 : x와 y가 모두 1인 수
$$SMC(A,B) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid} = \frac{f_{00} + f_{11}}{f_{00} + f_{01} + f_{10} + f_{11}}$$

###  Jaccard Distance
단순 매칭은 비대칭 이진 속성의 경우 문제가 발생(비대칭 이진 속성 : 0이 아닌 속성만 유효한 속성). 이런 경우 f00의 빈도가 매우 높기 때문에 f00빈도를 제거하고 유사도를 측정
$$J(A,B) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid} = \frac{f_{11}}{f_{01} + f_{10} + f_{11}}$$

### Cosine Similarity
문서간 유사도를 측정할 때 많이 사용되는 척도
$$cos(x,y) = \frac{x \cdot y}{\lVert x \rVert \lVert y \rVert}, \begin{cases} \text{완전 동일}, & \text{if cos(x,y)} = 1 \\ \text{완전 반대}, & \text{if cos(x,y)} = -1 \\ \text{독립}, & \text{if cos(x,y)} = 0 \end{cases} $$

### Hamming Distance
같은 길이를 가진 두 개의 문자열에서 같은 위치에 있지만 서로 다른 문자의 개수
* '1011101'과 '1001001'사이의 해밍 거리는 2이다. (1011101, 1001001)
* '2143896'과 '2233796'사이의 해밍 거리는 3이다. (2143896, 2233796)
* "toned"와 "roses"사이의 해밍 거리는 3이다. (toned, roses)

## 정규화(Normalization)
변수의 단위에 따라 거리에 미치는 영향력이 다르기 때문에 정규화를 통해 변수를 변환하여 사용
* 최소-최대 정규화 : 데이터의 최대/최소 값을 이용하여 모든 속성 값을 0과 1 사이의 값으로 변환
$$ Xnew = \frac{X - Xmin}{Xmax - Xmin}$$
* 표준화(Standardazation) : 데이터의 평균을 빼고 데이터의 표준 편차로 나누어 변환
$$ Xnew = \frac{X - Xmean}{Xsd}$$

## K
k-NN에서 얼마나 많은 이웃(K)을 사용할지 결정하는 것은 데이터에 의존적
여러가지 K값을 시도해보고 평가 척도가 가장 좋은 K값을 선택
* 너무 작은 k : Overfitting의 우려
* 너무 큰 k : Outlier의 영향이 줄어들지만, 항목간 경계가 불문명해짐(Underfitting)

## K-NN의 문제점들
* 차원의 저주 : 차원이 너무 많아서 최근접이웃이라 해도 현실적으로는 '가깝다'라고 하기엔 너무 멈
* 과적합 : 어떤 이웃이 갖아 가까워도 그 이웃이 완전히 잡음(Noise)일 수 있음(K를 증가시키면 잡음의 영향을 줄일 수 있음)
* 변수들간의 상관관계 : 변수들이 매우 많기 때문에 이들 간에 서로 높은 상관을 가진 변수가 있을 수 있음(상관에 대한 이해를 바탕으로 적은 차원의 공간으로 투영)
* 변수들의 상대적 중요성 : 어떤 변수들은 다른 변수보다 중요할 수 있음(변수에 가중치를 부여)
* 희소 : 벡터 또는 행렬이 희소

# R 코드 예제

### Import Libraies


```R
pkgs <- c("caret", "class", "kknn", "readxl")
sapply(pkgs, require, character.only = T)
```


<dl class=dl-horizontal>
	<dt>caret</dt>
		<dd>TRUE</dd>
	<dt>class</dt>
		<dd>TRUE</dd>
	<dt>readxl</dt>
		<dd>TRUE</dd>
</dl>



### Load Data


```R
churn <- read_excel("C:/Users/rymyu/Dropbox/Public/공부/Cheat Sheet/data/churn.xlsx", sheet=1, col_names=T)
print(head(churn))
str(churn)
```

    # A tibble: 6 x 12
      COLLEGE INCOME OVERAGE LEFT~  HOUSE HAND~ OVER~ AVER~ REPO~ REPO~ CONS~ LEAVE
      <chr>    <dbl>   <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl> <chr> <chr> <chr> <chr>
    1 zero     31953     0    6.00 313378   161  0     4.00 unsat litt~ no    STAY 
    2 one      36147     0   13.0  800586   244  0     6.00 unsat litt~ cons~ STAY 
    3 one      27273   230    0    305049   201 16.0  15.0  unsat very~ perh~ STAY 
    4 zero    120070    38.0 33.0  788235   780  3.00  2.00 unsat very~ cons~ LEAVE
    5 one      29215   208   85.0  224784   241 21.0   1.00 very~ litt~ neve~ STAY 
    6 zero    133728    64.0 48.0  632969   626  3.00  2.00 unsat high  no    STAY 
    Classes 'tbl_df', 'tbl' and 'data.frame':	20000 obs. of  12 variables:
     $ COLLEGE                    : chr  "zero" "one" "one" "zero" ...
     $ INCOME                     : num  31953 36147 27273 120070 29215 ...
     $ OVERAGE                    : num  0 0 230 38 208 64 224 0 0 174 ...
     $ LEFTOVER                   : num  6 13 0 33 85 48 0 20 7 18 ...
     $ HOUSE                      : num  313378 800586 305049 788235 224784 ...
     $ HANDSET_PRICE              : num  161 244 201 780 241 626 191 357 190 687 ...
     $ OVER_15MINS_CALLS_PER_MONTH: num  0 0 16 3 21 3 10 0 0 25 ...
     $ AVERAGE_CALL_DURATION      : num  4 6 15 2 1 2 5 5 5 4 ...
     $ REPORTED_SATISFACTION      : chr  "unsat" "unsat" "unsat" "unsat" ...
     $ REPORTED_USAGE_LEVEL       : chr  "little" "little" "very_little" "very_high" ...
     $ CONSIDERING_CHANGE_OF_PLAN : chr  "no" "considering" "perhaps" "considering" ...
     $ LEAVE                      : chr  "STAY" "STAY" "STAY" "LEAVE" ...
    

### Preprocessing


```R
churn_new <- churn
churn_new$COLLEGE <- as.factor(churn_new$COLLEGE)
idx <- c(2,3,4,5,6,7,8)
churn_new[idx] <- lapply(churn_new[idx], scale)
```

### Data Partition


```R
train_idx_new <- createDataPartition(churn_new$COLLEGE, p=0.7, list=F)
train_new <- churn_new[train_idx_new,]
test_new <- churn_new[-train_idx_new,]
print(dim(train_new))
print(dim(test_new))
```

    [1] 14001    12
    [1] 5999   12
    

### k-NN Implementation


```R
knn.pred <- knn(train_new[,c(2:8)], test_new[,c(2:8)], train_new$COLLEGE, k=3)
```


```R
confusionMatrix(knn.pred, test_new$COLLEGE, mode="prec_recall")
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction  one zero
          one  1526 1480
          zero 1488 1505
                                             
                   Accuracy : 0.5053         
                     95% CI : (0.4925, 0.518)
        No Information Rate : 0.5024         
        P-Value [Acc > NIR] : 0.3350         
                                             
                      Kappa : 0.0105         
     Mcnemar's Test P-Value : 0.8978         
                                             
                  Precision : 0.5077         
                     Recall : 0.5063         
                         F1 : 0.5070         
                 Prevalence : 0.5024         
             Detection Rate : 0.2544         
       Detection Prevalence : 0.5011         
          Balanced Accuracy : 0.5052         
                                             
           'Positive' Class : one            
                                             


###  Search K by *knn* Library


```R
ctrl <- trainControl(method="cv", number=10)
```


```R
knn.fit <- train(COLLEGE ~ INCOME + OVERAGE + LEFTOVER + HOUSE + HANDSET_PRICE + OVER_15MINS_CALLS_PER_MONTH + AVERAGE_CALL_DURATION,
                data = train, method="knn", trControl=ctrl, preProcess=c("center", "scale"), tuneLength=10)
```


```R
knn.fit
```


    k-Nearest Neighbors 
    
    14001 samples
        7 predictor
        2 classes: 'one', 'zero' 
    
    Pre-processing: centered (7), scaled (7) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 12600, 12601, 12602, 12601, 12601, 12600, ... 
    Resampling results across tuning parameters:
    
      k   Accuracy   Kappa     
       5  0.5093203  0.01860726
       7  0.5101792  0.02033258
       9  0.5124651  0.02486757
      11  0.5131078  0.02620234
      13  0.5203927  0.04076725
      15  0.5147500  0.02948654
      17  0.5128224  0.02560674
      19  0.5152503  0.03049859
      21  0.5099631  0.01988623
      23  0.5058917  0.01174017
    
    Accuracy was used to select the optimal model using  the largest value.
    The final value used for the model was k = 13.



```R
trellis.par.set(caretTheme())
plot(knn.fit)
```

![searchK](https://www.dropbox.com/s/gcan2w0ijbfnnns/knn3.jpg?raw=1)


```R
knn.pred2 <- predict(knn.fit, test)
```


```R
confusionMatrix(knn.pred2, test$COLLEGE, mode="prec_recall")
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction  one zero
          one  1543 1477
          zero 1471 1508
                                              
                   Accuracy : 0.5086          
                     95% CI : (0.4958, 0.5213)
        No Information Rate : 0.5024          
        P-Value [Acc > NIR] : 0.1730          
                                              
                      Kappa : 0.0171          
     Mcnemar's Test P-Value : 0.9266          
                                              
                  Precision : 0.5109          
                     Recall : 0.5119          
                         F1 : 0.5114          
                 Prevalence : 0.5024          
             Detection Rate : 0.2572          
       Detection Prevalence : 0.5034          
          Balanced Accuracy : 0.5086          
                                              
           'Positive' Class : one             
                                              


### search K by *kknn* Library


```R
knn.fit <- train(COLLEGE ~ INCOME + OVERAGE + LEFTOVER + HOUSE + HANDSET_PRICE + OVER_15MINS_CALLS_PER_MONTH + AVERAGE_CALL_DURATION,
                data = train, method="kknn", trControl=ctrl, preProcess=c("center", "scale"), tuneLength=10)
```


```R
knn.fit
```


    k-Nearest Neighbors 
    
    14001 samples
        7 predictor
        2 classes: 'one', 'zero' 
    
    Pre-processing: centered (7), scaled (7) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 12602, 12601, 12601, 12601, 12602, 12600, ... 
    Resampling results across tuning parameters:
    
      kmax  Accuracy   Kappa     
       5    0.5120336  0.02408037
       7    0.5125338  0.02508037
       9    0.5126052  0.02522381
      11    0.5106775  0.02135425
      13    0.5108918  0.02177864
      15    0.5138903  0.02778545
      17    0.5132474  0.02649338
      19    0.5133903  0.02674704
      21    0.5133903  0.02674704
      23    0.5133903  0.02674704
    
    Tuning parameter 'distance' was held constant at a value of 2
    Tuning
     parameter 'kernel' was held constant at a value of optimal
    Accuracy was used to select the optimal model using  the largest value.
    The final values used for the model were kmax = 15, distance = 2 and kernel
     = optimal.



```R
trellis.par.set(caretTheme())
plot(knn.fit)
```

![searchK2](https://www.dropbox.com/s/5zrc2k7y6y2chmi/knn4.jpg?raw=1)


```R
knn.pred2 <- predict(knn.fit, test)
```


```R
confusionMatrix(knn.pred2, test$COLLEGE, mode="prec_recall")
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction  one zero
          one  1517 1487
          zero 1497 1498
                                              
                   Accuracy : 0.5026          
                     95% CI : (0.4898, 0.5153)
        No Information Rate : 0.5024          
        P-Value [Acc > NIR] : 0.4949          
                                              
                      Kappa : 0.0052          
     Mcnemar's Test P-Value : 0.8691          
                                              
                  Precision : 0.5050          
                     Recall : 0.5033          
                         F1 : 0.5042          
                 Prevalence : 0.5024          
             Detection Rate : 0.2529          
       Detection Prevalence : 0.5008          
          Balanced Accuracy : 0.5026          
                                              
           'Positive' Class : one             
                                              

