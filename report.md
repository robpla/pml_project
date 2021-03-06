# Practical Machine Learning assignment
## 1. Problem overview
The problem to solve is to correctly classify the way a few persons using devices such as Jawbone Up, Nike FuelBand, and Fitbit perform their training activities. The activities fall into one of the below classes:  
* A: exactly according to the specification,  
* B: throwing the elbows to the front,  
* C: lifting the dumbbell only halfway,  
* D: lowering the dumbbell only halfway,  
* E: throwing the hips to the front  
Data was recorded from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.  
I aim for accuracy greater than 99.5%. 

## 2. Models overview

Due to the nature of the problem, classification algorithms were applied. Below is a table with the algorithms, cross validation accuracies\* (accuracy = 1 - out_of_sample_error), type of cross validation and a brief comment. The cross-validated model with the **best accuracy** on the cross validation set is in **bold** font. This algorithm was later applied to the test data.

\* the original data set was split in 60%-20%-20% manner between training, cross validation and test sets, respectively

### 2.a. Training and cross validation data set accuracy estimation


No. | Algorithm        | CV accuracy | CV used | Comments
----|------------------|-------------|---------|---------
1.  | single tree1     | 0.6721 | 2-fold | all persons are predicted with a single tree (i.e. one confusion matrix); this results remains roughly the same whether or not the *user_name* variable is included in the predictors;  10-fold cross validation used for tree pruning, however the most complex tree was chosen by the algorithm, hence no pruning applied
2.  | single tree2     | 0.9782 | 2-fold | each person is predicted with an individual tree, and the overall accuracy of all 6 trees is presented (i.e. six confusion matrices are summed together and the overall result is presented); 10-fold cross validation used for tree pruning, however the most complex tree was chosen by the algorithm, hence no pruning applied
3.  | bagged tree1     |0.9562 | 2-fold  | all persons are predicted with a model with 6 trees only
4.  | random forest1   | 0.9602 | 2-fold | all persons are predicted with a single random forest of 6 trees, 7 variables per node (i.e. approximate half of the number of predictors)
5.  | random forest2   | 0.9852 | 2-fold   | all persons are predicted with a single random forest of 500 trees, 7 variables per node (i.e. approximate of the square root of the number of predictors)
6.  | random forest3   | 0.9831 | 2-fold | all persons are predicted with a single random forest of 500 trees, 27 variables per node (i.e. approximate half of the number of predictors)
7. | random forest4 | 0.9858   | 2-fold | all persons are predicted with a single random forest of 500 trees, 3 variables per node
**8.**  | **bagged tree2** | **0.9977** | **2-fold** | **all persons are predicted with a model with 500 trees**


## 3. Models building

20% of the original data was held out as cross validation data. Additional 20% was held out as test data and was not used for model learning. Standard data cleaning and partitioning was done. Only top two cross validated models with good accuracy will be presented in this section.

### 3.a. Data cleaning

Columns with more than 50% missing data were removed from the training and cross validation set. Additionally, id (*V1*), windows and timestamps data were removed.

```r
data <- fread("pml-training.csv", na.strings=c("NA","","#DIV/0!"))
skip_cols <- sapply(data, function(dt) sum(is.na(dt))/length(dt) > 0.5)
data <- data[,!skip_cols]
data <- data[, !names(data) %in% c("V1", "num_window", "new_window", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp")]
```

### 3.b. Creating data partitions

60% of the original data was used to create the training set, 20% - the cross validation set and last 20% - the test set. Seed of *1234* was used when creating the data partitions. Test data was not used for model learning.

```r
set.seed(1234)
in_trainig <- createDataPartition(data$classe,p=0.6,list = FALSE)
training <- data[in_trainig,] ## training set
validation_test <- data[-in_trainig,]
in_validation <- createDataPartition(validation_test$classe,p=0.5,list = FALSE)
validation <- validation_test[in_validation,] ## cross validation set
test <- validation_test[-in_validation,] ## test set
```

### 3.c. Models

Only top two models are presented below.  
1. Bagged tree2

```r
## fold1
bagged2 <- randomForest(classe~.,data = training, mtry=length(training)-1, ntree=500, importance=T)
bagged2.pred <- predict(bagged2,newdata=validation)
#### fold2
bagged2.f2 <- randomForest(classe~.,data = validation, mtry=length(validation)-1, ntree=500, importance=T)
bagged2.f2.pred <- predict(bagged2,newdata=training)
## cv accuracy = 0.9976557
sum(diag(table(bagged2.pred,validation$classe)),diag(table(bagged2.f2.pred,training$classe)))/sum(nrow(training),nrow(validation))
```
2. Random forest4

```r
## fold1
rf4 <- randomForest(classe~.,data = training, mtry=3, ntree=500, importance=T)
rf4.pred <- predict(rf4,newdata=validation)
## fold2
rf4.f2 <- randomForest(classe~.,data = validation, mtry=3, ntree=500, importance=T)
rf4.f2.pred <- predict(rf4.f2,newdata=training)
## cv accuracy = 0.9857813
sum(diag(table(rf4.pred,validation$classe)),diag(table(rf4.f2.pred,training$classe)))/sum(nrow(validation),nrow(training))
```


## 4. Use of cross validation
Due to computational reasons, 2-fold cross validation (i.e. holdout) was applied to estimate accuracy in the cross validation step. 

## 5. Expected out of sample error
The expected out of sample error is approx. 1.50% per the test set.

```r
test_prediction <- predict(bagged2,newdata=test)
1-sum(diag(table(test_prediction,test$classe)))/nrow(test)
## error = 0.0150395
```
