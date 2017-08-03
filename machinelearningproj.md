# Regression Models Course Project
Phil Bagienski  
August 2, 2017  
#Overview
The goal of this project is the predict the manner in which a group of 6 participants performed barbell lifts correctly and incorrectly in 5 different ways. The data comes from accelerometers on the belt, forearm, arm, and dumbell from this source: http://groupware.les.inf.puc-rio.br/har. I will be applying machine learning techniques to build a prediction model and use it to predict 20 different test cases. In particular, I will use a random forest model.



#Loading the data
First let's load up the data and take a general glance at it. Make sure the testing and training sets have been downloaded to the desired working directory.


```r
#loading necessary libraries for later
library(caret)
library(randomForest)

#set seed for reproducibility
set.seed(123)
```

```r
#read CSV files
train <- read.csv("pml-training.csv", header=T,sep=",",na.strings=c("NA",""))
test <- read.csv("pml-testing.csv", header=T,sep=",",na.strings=c("NA",""))

#look at dimensions
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

#Data Cleaning
We now know that there are quite a bit of columns, ie possible predictors, for our exploration. By using head() or names() to look at the data, clearly some are irrelevant to our model (ie, ID), so let's first remove them. In particular, the first seven column names do not appear to be relevant for both datasets.


```r
##subset only potentially relevant predictors
newTrain <- train[,8:160]
newTest <- test[,8:160]
```

Now we have to account for the fact that many of these potential predictors have a lot of NA values. Since this could mess up our model, we will remove any columns with less than 60% data.


```r
#find number of columns with less than 60% data
sum((colSums(!is.na(newTrain[,-ncol(newTrain)])) < 0.6*nrow(newTrain)))
```

```
## [1] 100
```

```r
## 100 columns are missing a lot of data! Lets remove these.
satisfactory <- c((colSums(!is.na(newTrain[,-ncol(newTrain)])) >= 0.6*nrow(newTrain)))

newTrain <- newTrain[,satisfactory]
newTest <- newTest[,satisfactory]

dim(newTrain)
```

```
## [1] 19622    53
```

```r
dim(newTest)
```

```
## [1] 20 53
```

We have reduced our predictors from 160 down to 53. This should be good enough to proceed to modeling.

#Model Building
Now we are ready to model. First lets create a partition within our training set so that we have data set aside for validation. We will allow 60% data to be used as creating our model and 40% for validation.


```r
inTrain = createDataPartition(newTrain$classe, p=0.60, list=F)
training = newTrain[inTrain,]
validating = newTrain[-inTrain,]
```

Good, now we can create our model using the randomForest function. In this situation, classe will be our dependent value which we are trying to predict.




```r
#this may take awhile
model <- randomForest(classe~.,data=training)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    2    0    1    2 0.001493429
## B   11 2263    5    0    0 0.007020623
## C    0   19 2032    3    0 0.010710808
## D    1    0   23 1905    1 0.012953368
## E    0    0    2    7 2156 0.004157044
```

#Validating the model

Almost done. Now lets see how our model performs on the validation data we set aside from the partition!

```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    9    0    0    0
##          B    3 1506   11    0    0
##          C    0    3 1355   11    3
##          D    0    0    2 1275    2
##          E    0    0    0    0 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9925, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9921   0.9905   0.9914   0.9965
## Specificity            0.9984   0.9978   0.9974   0.9994   1.0000
## Pos Pred Value         0.9960   0.9908   0.9876   0.9969   1.0000
## Neg Pred Value         0.9995   0.9981   0.9980   0.9983   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1919   0.1727   0.1625   0.1832
## Detection Prevalence   0.2852   0.1937   0.1749   0.1630   0.1832
## Balanced Accuracy      0.9985   0.9949   0.9939   0.9954   0.9983
```
Looks pretty good. Finally, we can use the accuracy from this to estimate the out of sample error rate.

```r
accuracy<-c(as.numeric(predict(model,newdata=validating[,-ncol(validating)])==validating$classe))
accpercent<-sum(accuracy)*100/nrow(validating)
error<-100-accpercent
error
```

```
## [1] 0.5607953
```
Our estimated out of sample error rate is approx. .56%

#Testing on Test set
To conclude I shall run the model on the test set to (hopefully) get the right classifications for the given test set. We already transformed it in align with the training set's dimensions, so this should be pretty easy.


```r
predictions<-predict(model,newdata=newTest)
predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

#Conclusion
I consider this project a success in that the random forest model was quite accurate in the validation set. Some things that could improve the model could be: removing all columns with near zero variance and removing all columns which have a high correlation to other variables. This model accurately predicted the 20 cases in the prediction quiz.
