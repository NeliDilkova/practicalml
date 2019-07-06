---
title: "Practical Machine Learning Assignment"
author: "Tim Kuttig"
date: "6 July 2019"
output: 
  html_document: 
    keep_md: yes
---



## Summary

The aim of this assignment is to train a machine learning algorithm to predict the quality of execution of weight lifting excercises. The data used contains a total of 160 variables, most of which are taken from several sensors on the test subjects. The outcome variable comprises 5 different levels.
A random forest model is used for this assignment. Both in-sample and out-of-sample accuracy are very good at around 99% accuracy.

## Data

### Background

The dataset is taken from a study by Velloso et. al (2013) on the recognition of the quality of execution of weight lifting exercises. The authors provide the following description (taken from the [Human Activity Recognition Project website](http://groupware.les.inf.puc-rio.br/har)):

> This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.
> 
In this work (see the paper) we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach (dataset here), but also an "ambient sensing approach" (by using Microsoft Kinect - dataset still unavailable)
>
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
>
Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

### Examining and Processing the Data

The csv files containing the training and test data are accessed using the links provided by Coursera. Before exploring the data I split the provided dataset into a training and a test set. The model will only be trained on the training set so I can properly assess the out-of-sample prediction accuracy later on.


```r
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)

set.seed(672019)

# training - for analysis
pmlTraining <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na = c("", "NA", "#DIV/0!"))
# test - for prediction assignment
pmlTesting <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na = c("", "NA", "#DIV/0!"))

# split data into training and test set
inTrain  <- createDataPartition(pmlTraining$classe, p=0.75, list=FALSE)
trainSet <- pmlTraining[inTrain, ]
testSet  <- pmlTraining[-inTrain, ]
```

75% of the data are randomly assigned to the training set. This gives us a training set with 14718 and a test set with 4904 observations.


```r
str(trainSet)
```

```
## Classes 'tbl_df', 'tbl' and 'data.frame':	14718 obs. of  160 variables:
##  $ X1                      : num  1 2 4 5 6 7 8 9 10 11 ...
##  $ user_name               : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1    : num  1.32e+09 1.32e+09 1.32e+09 1.32e+09 1.32e+09 ...
##  $ raw_timestamp_part_2    : num  788290 808298 120339 196328 304277 ...
##  $ cvtd_timestamp          : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window              : chr  "no" "no" "no" "no" ...
##  $ num_window              : num  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.48 1.48 1.45 1.42 1.42 1.43 1.45 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 8.18 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : num  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.03 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0.02 0 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 ...
##  $ accel_belt_x            : num  -21 -22 -22 -21 -21 -22 -22 -20 -21 -21 ...
##  $ accel_belt_y            : num  4 4 3 2 4 3 4 2 4 2 ...
##  $ accel_belt_z            : num  22 22 21 24 21 21 21 24 22 23 ...
##  $ magnet_belt_x           : num  -3 -7 -6 -6 0 -4 -2 1 -3 -5 ...
##  $ magnet_belt_y           : num  599 608 604 600 603 599 603 602 609 596 ...
##  $ magnet_belt_z           : num  -313 -311 -310 -302 -312 -311 -313 -312 -308 -317 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 21.5 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : num  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0 0.02 0 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 0 ...
##  $ accel_arm_x             : num  -288 -290 -289 -289 -289 -289 -289 -288 -288 -290 ...
##  $ accel_arm_y             : num  109 110 111 111 111 111 111 109 110 110 ...
##  $ accel_arm_z             : num  -123 -125 -123 -123 -122 -125 -124 -122 -124 -123 ...
##  $ magnet_arm_x            : num  -368 -369 -372 -374 -369 -373 -372 -369 -376 -366 ...
##  $ magnet_arm_y            : num  337 337 344 337 342 336 338 341 334 339 ...
##  $ magnet_arm_z            : num  516 513 512 506 513 509 510 518 516 509 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 13.4 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.4 -70.4 -70.8 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -84.9 -84.9 -84.5 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

A first look at the data suggests that there are some variables that should be dropped before training the model, as they contain information that is not useful for predicting the quality of the workout (such as the ID in column 1, user_name, raw_timestamp, etc.). Furthermore, other variables on different sensor readings contain a lot of missing values (`NA`). Many machine learning algorithms are not made to handle variables with many missing values or very low variance, which is why such variables are oftentimes discarded.


```r
# identify columns with more than 95% missing values
mostlyNA <- sapply(trainSet, function(x) mean(is.na(x))) > 0.95
sum(mostlyNA==TRUE) # number of identified columns
```

```
## [1] 100
```

```r
# store variables to drop in vector so they can also be dropped from the test set
dropVars <- mostlyNA
dropVars[1:7] <- TRUE # drop ID, user_name etc.

# drop identified columns
trainSet <- trainSet[, !dropVars]
```

After dropping variables with more than 95% missing values or which are otherwise unsuitable, we are left with 53 variables, one of which is the variable `classe` that we want to predict. Using the `nearZeroVar` function from the `caret` package we can see that none of the remaining 52 variables seem to be problematic in terms of missing values or low variance.


```r
nzv <- nearZeroVar(select(trainSet, -classe), saveMetrics = TRUE)
nzv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.145706     7.7999728   FALSE FALSE
## pitch_belt            1.000000    11.5436880   FALSE FALSE
## yaw_belt              1.100515    12.3318386   FALSE FALSE
## total_accel_belt      1.065160     0.1902432   FALSE FALSE
## gyros_belt_x          1.050000     0.8764778   FALSE FALSE
## gyros_belt_y          1.146006     0.4620193   FALSE FALSE
## gyros_belt_z          1.071752     1.1210762   FALSE FALSE
## accel_belt_x          1.056338     1.0735154   FALSE FALSE
## accel_belt_y          1.151142     0.9308330   FALSE FALSE
## accel_belt_z          1.070465     1.9567876   FALSE FALSE
## magnet_belt_x         1.135849     2.0722924   FALSE FALSE
## magnet_belt_y         1.043564     1.9567876   FALSE FALSE
## magnet_belt_z         1.037143     2.9623590   FALSE FALSE
## roll_arm             49.230769    16.7142275   FALSE FALSE
## pitch_arm            73.171429    19.3708384   FALSE FALSE
## yaw_arm              30.117647    18.2973230   FALSE FALSE
## total_accel_arm       1.020438     0.4484305   FALSE FALSE
## gyros_arm_x           1.065753     4.2600897   FALSE FALSE
## gyros_arm_y           1.376289     2.4935453   FALSE FALSE
## gyros_arm_z           1.081425     1.5898899   FALSE FALSE
## accel_arm_x           1.048000     5.2248947   FALSE FALSE
## accel_arm_y           1.107143     3.6146216   FALSE FALSE
## accel_arm_z           1.135417     5.2588667   FALSE FALSE
## magnet_arm_x          1.073529     8.9754043   FALSE FALSE
## magnet_arm_y          1.042857     5.8499796   FALSE FALSE
## magnet_arm_z          1.011628     8.5473570   FALSE FALSE
## roll_dumbbell         1.009804    86.2413371   FALSE FALSE
## pitch_dumbbell        2.235294    83.8768854   FALSE FALSE
## yaw_dumbbell          1.146067    85.6162522   FALSE FALSE
## total_accel_dumbbell  1.061047     0.2921593   FALSE FALSE
## gyros_dumbbell_x      1.045752     1.6170675   FALSE FALSE
## gyros_dumbbell_y      1.263761     1.8616660   FALSE FALSE
## gyros_dumbbell_z      1.041096     1.3588803   FALSE FALSE
## accel_dumbbell_x      1.008000     2.7924990   FALSE FALSE
## accel_dumbbell_y      1.021390     3.0846582   FALSE FALSE
## accel_dumbbell_z      1.005076     2.7585270   FALSE FALSE
## magnet_dumbbell_x     1.097561     7.3311591   FALSE FALSE
## magnet_dumbbell_y     1.143939     5.6529420   FALSE FALSE
## magnet_dumbbell_z     1.033557     4.4978937   FALSE FALSE
## roll_forearm         11.232558    13.1403723   FALSE FALSE
## pitch_forearm        69.000000    18.2769398   FALSE FALSE
## yaw_forearm          15.328042    12.3997826   FALSE FALSE
## total_accel_forearm   1.110160     0.4688137   FALSE FALSE
## gyros_forearm_x       1.108179     1.9364044   FALSE FALSE
## gyros_forearm_y       1.035714     4.9191466   FALSE FALSE
## gyros_forearm_z       1.154971     2.0247316   FALSE FALSE
## accel_forearm_x       1.045455     5.3336051   FALSE FALSE
## accel_forearm_y       1.051948     6.6924854   FALSE FALSE
## accel_forearm_z       1.000000     3.8116592   FALSE FALSE
## magnet_forearm_x      1.067797     9.9741813   FALSE FALSE
## magnet_forearm_y      1.114754    12.5220818   FALSE FALSE
## magnet_forearm_z      1.000000    11.1156407   FALSE FALSE
```

The correlation matrix below shows that some of the remaining variables are highly correlated. This could be addressed by reducing dimensionality through Principal Component Analysis should the accuracy of our model be unsatisfactory.


```r
corMat <- select(trainSet, -classe) %>% cor()
corrplot(corMat, order = "FPC", method = "color", type = "lower", tl.cex = 0.7, tl.col = rgb(0, 0, 0))
```

![](index_files/figure-html/correlation-1.png)<!-- -->

## Model

I choose the Random Forest approach to train a model due to its generally high prediction accuracy. The model is trained using all 52 predictors. The variable importance will be checked later on. The `trainControl` function is set to apply 5-fold cross valiation in order to find a balance between bias and variance.

### Train Model


```r
# setup parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

control <- trainControl(method = "cv", number = 5, verboseIter=FALSE, allowParallel = TRUE)
model <- train(classe ~ ., data = trainSet, method = "rf", trControl = control)

# shut down cluster
stopCluster(cluster)
registerDoSEQ()

model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.58%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4179    5    1    0    0 0.001433692
## B   16 2829    3    0    0 0.006671348
## C    0   15 2550    2    0 0.006622517
## D    0    0   37 2373    2 0.016169154
## E    0    0    0    5 2701 0.001847746
```

```r
model$resample$Accuracy
```

```
## [1] 0.9881074 0.9901495 0.9945615 0.9918506 0.9904891
```

The model has a very good in-sample performance, with an estimated error rate of 0.58% when predicting `classe`. Accordingly, the accuracy of the k-fold cross validation is around 99% for all 5 iterations.

### Variable Importance


```r
modelImp <- varImp(model)

modelImp$importance %>%
    as.data.frame() %>%
    rownames_to_column() %>%
    arrange(Overall) %>%
    mutate(rowname = forcats::fct_inorder(rowname)) %>%
    rename(Feature = rowname, Importance = Overall) %>%
    ggplot() +
    geom_col(aes(x = Feature, y = Importance, fill = Importance)) +
    scale_fill_viridis_c(option = "D", direction = -1) +
    coord_flip() +
    theme_bw()
```

![](index_files/figure-html/var_imp-1.png)<!-- -->

The plot shows that the roll, yaw and pitch measured on the belt are among the top predictors for the `classe` variable, while variables with low importance could probably be dropped from the dataset without too much of an impact on prediction accuracy.

### Apply Model to Test Set

Now that the model is trained and shows good in-sample accuracy it's time to apply it to the test set to check out-of-sample performance. Variables that have been dropped from the training set earlier are also dropped from the test set.


```r
# drop variables from test set
testSet <- testSet[, !dropVars]

# apply model to predict classe
prediction <- predict(model, newdata = testSet)
results <- confusionMatrix(prediction, as.factor(testSet$classe))
results
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  944    8    0    0
##          C    0    1  845   15    0
##          D    0    0    2  789    3
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9933          
##                  95% CI : (0.9906, 0.9954)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9883   0.9813   0.9967
## Specificity            0.9989   0.9980   0.9960   0.9988   1.0000
## Pos Pred Value         0.9971   0.9916   0.9814   0.9937   1.0000
## Neg Pred Value         1.0000   0.9987   0.9975   0.9964   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1925   0.1723   0.1609   0.1831
## Detection Prevalence   0.2853   0.1941   0.1756   0.1619   0.1831
## Balanced Accuracy      0.9994   0.9964   0.9922   0.9901   0.9983
```

The out-of-sample accuracy of the trained model is very good. The prediction accuracy lies above 99% when applied to the test set.

### Prediction Quiz

The model can be applied to the provided quiz test set with the following code (not run to adhere to the Coursera Honour Code).


```r
# drop variables from quiz test set
quizSet <- pmlTesting[, !dropVars]

# apply model to predict classe
prediction2 <- predict(model, newdata = select(quizSet, -problem_id))
data.frame(problem_id = quizSet$problem_id, prediction = prediction2)
```

## References

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 
