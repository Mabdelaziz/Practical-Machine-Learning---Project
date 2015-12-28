# Barbell lifts Classification - Project
Mohamed Abdelaziz Hassan Galal  
Sunday, December 27, 2015  

## Introduction 

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

# Analysis

## 1. Set the Environemnt and Load data Set of training and testing the ML model.


```r
library(AppliedPredictiveModeling)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library("rpart")
library("rpart.plot")
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed("1234")
```



```r
fname.training <- "pml-training.csv"
fname.testing <- "pml-testing.csv"
if ( !file.exists(fname.training))
  {
  trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(trainUrl,fname.training, method="auto" )  
  }
if (!file.exists(fname.testing) )
{
  testUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(testUrl,fname.testing, method="auto" )   
}

# read data into R data frames.
df_training <- read.csv(fname.training, na.strings=c("NA","","<NA>"), header=TRUE)
cnames_train <- colnames(df_training)
df_testing <- read.csv(fname.testing, na.strings=c("NA","","<NA>"), header=TRUE)
cnames_test <- colnames(df_testing)
```

##2. Preprocessing training data.

Here we are aiming to remove NAs columns from training data sets. In order to not lose an important feature we set condition on Na ratio which greater than 50% of the training data sample.


```r
nacols<- c(1,2,3,4,5) ## X ,UserName,and Timestamps  Attributes hav no meananing with Prediction Model
training.size<- nrow (df_training)
for( i in 1 : ncol(df_training))
{
nasum<- sum(is.na(df_training[,i]))
if (nasum/training.size>.5) { nacols<- cbind( nacols, i) }
}


df_trainv2<- df_training[,-nacols] 
features_cnt<- ncol(df_trainv2)-1
df_testingv2 <- df_testing[,-nacols]
```

After removing NA columns which has NA ratio > 50% we currently have 54 features.
We will check if these features contain any Nulls to be mutated with nearest k means values, if any!


```r
sum(is.na(df_trainv2))
```

```
## [1] 0
```

```r
DataNZV <- nearZeroVar(df_trainv2, saveMetrics=TRUE)
colnames(df_trainv2)
```

```
##  [1] "new_window"           "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [55] "classe"
```

```r
df_train<- df_trainv2
```

Features we counted have no missing values. And we can check the variability of each feature using nearZeroVar analysis.

## Exploratory analysis
The variable classe contains 5 levels. The plot of the outcome variable shows the frequency of each levels in the subTraining data.


```r
g<- ggplot (df_train , aes(classe) )
g<- g + geom_bar( aes(fill=classe))
g<- g + ggtitle("Classes Frequencies with in Training Data Set" )
g
```

![](BarBell_lifts_clssification_files/figure-html/unnamed-chunk-5-1.png) 

## 3. Data Partitioning


```r
subindx <- createDataPartition(y =df_train$classe, p=0.75, list=FALSE)
subTraining <- df_train[subindx, ] 
subCV <- df_train[-subindx, ]
```

** Apply Decision Tree model**


```r
FitDT <- rpart(classe ~ ., data=subTraining, method="class")

# Perform prediction
predictDT <- predict(FitDT, subCV, type = "class")
resDT= (subCV$class==predictDT ) 
# Plot result
rpart.plot(FitDT, main="Decision Tree")
```

![](BarBell_lifts_clssification_files/figure-html/unnamed-chunk-7-1.png) 

```r
confusionMatrix(predictDT, subCV$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1234  171   33   65   51
##          B   58  584   53   65  104
##          C   10   60  686  111   75
##          D   75  101   57  520  102
##          E   18   33   26   43  569
## 
## Overall Statistics
##                                        
##                Accuracy : 0.7327       
##                  95% CI : (0.72, 0.745)
##     No Information Rate : 0.2845       
##     P-Value [Acc > NIR] : < 2.2e-16    
##                                        
##                   Kappa : 0.6607       
##  Mcnemar's Test P-Value : < 2.2e-16    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8846   0.6154   0.8023   0.6468   0.6315
## Specificity            0.9088   0.9292   0.9368   0.9183   0.9700
## Pos Pred Value         0.7941   0.6759   0.7282   0.6082   0.8258
## Neg Pred Value         0.9519   0.9097   0.9573   0.9299   0.9212
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2516   0.1191   0.1399   0.1060   0.1160
## Detection Prevalence   0.3169   0.1762   0.1921   0.1743   0.1405
## Balanced Accuracy      0.8967   0.7723   0.8696   0.7825   0.8008
```

Error of the predection using DT model 


```r
perDT<- table ( resDT)
errorDT <- perDT[1]/perDT[2]
perDT
```

```
## resDT
## FALSE  TRUE 
##  1311  3593
```

we get error using decision tree model is about36.4876148%.

** Apply random Forest model**


```r
FitRF <- randomForest(classe ~ ., data=subTraining, method="class")
predictRF <- predict(FitRF, subCV[,-ncol(subCV)], type = "class")
resRF= (subCV$class==predictRF ) 
confusionMatrix(predictRF, subCV$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  948    5    0    0
##          C    0    0  850    5    0
##          D    0    0    0  799    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9978         
##                  95% CI : (0.996, 0.9989)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9972         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9989   0.9942   0.9938   1.0000
## Specificity            0.9997   0.9987   0.9988   1.0000   1.0000
## Pos Pred Value         0.9993   0.9948   0.9942   1.0000   1.0000
## Neg Pred Value         1.0000   0.9997   0.9988   0.9988   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1933   0.1733   0.1629   0.1837
## Detection Prevalence   0.2847   0.1943   0.1743   0.1629   0.1837
## Balanced Accuracy      0.9999   0.9988   0.9965   0.9969   1.0000
```

Error of the predection using random Forest model:


```r
perRF<- table ( resRF)
errorRF <- perRF[1]/perRF[2]
perRF
```

```
## resRF
## FALSE  TRUE 
##    11  4893
```

we get error using random Forest model is about0.224811%.


## Choose Model and apply on test data set

Form above model performance we select random Forest model to be applied on test data set.


```r
df_test<- df_testingv2 
# Perform prediction on test 
predictTest <- predict(FitDT, df_test[,-ncol(df_test)], type = "class")
```

# Write the output of the prediction model

```r
result <- data.frame(cbind (df_test[,ncol(df_test)] , levels(predictTest) ),stringsAsFactors=T)
colnames(result)<- c("problem_id" , "Predicted_Calss")
write.table((result)  ,file ="AllTestCases.txt" ,row.names=FALSE )

# write file per test case on output directory
write_files = function(results){
  for(i in 1:length(results)){
    filename = paste0("submission\\problem_id_",i,".txt")
    write.table(results[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
write_files(predictTest)
```




