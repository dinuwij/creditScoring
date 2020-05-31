
#######################################################
#**********ASSIGNMENT :CREDIT SCORING ***************#
#*********** ST: NAME: DINU WIJAYAWEERA *************#
#*********** PROF: Prof Dr. Wouter Verbeke  *********#
######################################################

#### References : 
# * Statistics and Machine Learning for marketing notes
# * Blue courses Basic Credit Risk Modeling for Basel/IFRS 9 using R/Python/SAS: https://www.bluecourses.com/courses/
# * Introduction to Credit scoring and scorecard : https://rpubs.com/chidungkt/442168
# * Survival Analysis : https://www.r-bloggers.com/steps-to-perform-survival-analysis-in-r/
# * Score card Module : https://cran.r-project.org/web/packages/scorecard/scorecard.pdf


# Data processing library
library(data.table)       # Data manipulation
library(readxl)             #Excel 
library(plyr)             # Data manipulation
library(dataPreparation)  # Data preparation library
library(woeBinning)       # Decision treebased binning for numerical and categorical variables
library(dplyr)            # Data Manipulation
library(scorecard)       # Specific for score card creation

# Machine learning library
library(mlr)          # Machine learning framework
library(caret)         # Data processing and machine learning framework

#install.packages("devtools", dependencies = TRUE)
#library(devtools)
#remotes::update_packages("rlang")


#clear working environment
rm(list=ls())

#Set working directory
setwd("C:/Users/dwijayaweera/Documents/2nd-Credit Scoring") 


#Read in the two data files
accepts <- read_excel("ACCEPTS.xlsx") 
rejects <- read_excel("REJECTS.xlsx") 

str(accepts)

#Change name of Default
names(accepts)[19] <- "Bad_creditability"
colnames(accepts)

# Check missing value
apply(is.na(accepts), 2, sum)

# Apply zero for missing values FICO Score assuming there is no scoring given for the particular customer
accepts$FICO_Score[is.na(accepts$FICO_Score)] <- 0

# check if there are missing values with Has_FICO condition: subset(accepts, Has_FICO==1 & FICO_Score==0)

#Convert to appropriate data types for better performance of the model
accepts$V8<-as.factor(accepts$V8)
accepts$Business_channel<-as.factor(accepts$Business_channel)
accepts$Bad_creditability<-as.factor(Bad_creditability)

accepts$FICO_Score<-as.double(accepts$FICO_Score)
accepts$Loan_amount<-as.double(accepts$Loan_amount)
accepts$Monthly_income<-as.double(accepts$Monthly_income)
accepts$Age<-as.double(accepts$Age)
accepts$Gearing_coefficient<-as.double(accepts$Gearing_coefficient)
accepts$Max_gearing_ratio<-as.double(accepts$Max_gearing_ratio)

#checking the changes
accepts$Business_channel<-as.factor(accepts$Business_channel)
str(accepts)
tail(accepts)


#Drop Days_late

accepts<-accepts[-c(12)]
head(accepts)


#Investigate distribution of the FICO Score
hist(accepts$FICO_Score)

#Investigate distribution of loan amount
hist(accepts$Loan_amount)

#Check the popular business channels 1 = via branch of financial institution, 2 = via partner retailer, 3 = broker
plot(accepts$Business_channel)

#Compare channels with creditability where 1 = bad

plot(accepts$Business_channel, accepts$Bad_creditability)

#There are outliers in FICO Score yet we assume that they are correct as it is provided by responsible instituitions
boxplot(accepts$FICO_Score)

# Check if there are outliers in age
boxplot(accepts$Age)

#Plot the data to explore the pattern of the data
plot(accepts,
  main="Loan accepts",
  col="dark green")

# Plot the target variable of bad creditability 
plot(as.factor(accepts$Bad_creditability), breaks=2, 
     xlab="Good (0) and Bad (1)", col="blue")

accepts$Default<-as.numeric(accepts$Default)

str(rejects)

# Check missing value
apply(is.na(rejects), 2, sum)

# Apply zero for missing values FICO Score assuming there is no scoring given for the particular customer
rejects$FICO_Score[is.na(rejects$FICO_Score)] <- 0

#Convert to appropriate data types for better performance of the model
rejects$V8<-as.factor(rejects$V8)
rejects$FICO_Score<-as.double(rejects$FICO_Score)
rejects$Loan_amount<-as.double(rejects$Loan_amount)
rejects$Monthly_income<-as.double(rejects$Monthly_income)
rejects$Age<-as.double(rejects$Age)
rejects$Gearing_coefficient<-as.double(rejects$Gearing_coefficient)
rejects$Max_gearing_ratio<-as.double(rejects$Max_gearing_ratio)

#Split the data for training and testing

set.seed(1001)
train_idx <- caret::createDataPartition(y=accepts$Bad_creditability, p=.7, list=F)
train <- accepts[train_idx, ]  # Train 70%
valid <- accepts[-train_idx, ] # Valid (holdout) 30%




# Checking the percentage distribution of target variable in each set
table(train$Bad_creditability)/nrow(train)

table(valid$Bad_creditability)/nrow(valid)

# Train and Valid sets are assigned with multiple classes hence need to bring them both to data.frame class
train<-as.data.frame(train)
valid<-as.data.frame(valid)
rejects<-as.data.frame(rejects)
class(valid) #check if it is rightfully converted

library(woeBinning)

# Grouping 
#binning_cat <- woe.binning(train, 'Bad_creditability','Business_channel')

#binning_cat

# Apply the binning to data
#tmp <- woe.binning.deploy(train, binning_cat, add.woe.or.dum.var='woe')
#head(tmp[, c('Business_channel', 'Business_channel.binned', 'woe.Business_channel.binned')])

subset=select(train,-c(Bad_creditability,ID))
tempbin<-woe.binning(train, 'Bad_creditability',subset)



tempbin

# Plot the binned variables
woe.binning.plot(tempbin)


# Tabulate the binned variables
tabulate.binning <- woe.binning.table(tempbin)
tabulate.binning


subs<-train

# Deploy the binning solution to the data frame
# (i.e. add binned variables and corresponding WOE variables)
train_woe <- woe.binning.deploy(subs, tempbin,
                                               add.woe.or.dum.var='woe')
# }

# Apply binned values on valid
valid_woe <- woe.binning.deploy(valid, tempbin,
                                               add.woe.or.dum.var='woe')

# Apply binned values on rejects
rejects_woe <- woe.binning.deploy(rejects, tempbin, add.woe.or.dum.var='woe')


colnames(rejects_woe)
colnames(train_woe)


head(iv(train_woe,"Bad_creditability",order=TRUE))

# Get the IV and DV list name
# Dependent variable (DV)
dv_list <- c('Bad_creditability')
# Independent variable (IV)
iv_list <- setdiff(colnames(train_woe), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'ID')  # Exclude the client_id

# Pick out categorical, boolean and numerical variable
iv_cat_list <- c()  # List to store categorical variable
iv_bool_list <- c()  # List to store boolean variable
iv_num_list <- c()  # List to store numerical variable
for (v in iv_list) {
    if (class(train_woe[, v]) == 'factor') {  # Factor == categorical variable
        iv_cat_list <- c(iv_cat_list, v)
    } else if (class(train_woe[, v]) == 'logical') {  # Logical == boolean variable
        iv_bool_list <- c(iv_bool_list, v)
    } else {  # Non-factor + Non-logical == numerical variable
        iv_num_list <- c(iv_num_list, v)
    }
}

#Drop categorical data as all were processed

for (v in iv_cat_list) {
    # Train, valid, test
    train_woe[, v] <- NULL
    valid_woe[, v] <- NULL
   
    
    # rejects 
    rejects_woe[, v] <- NULL
}

# Convert boolean to numeric
for (v in iv_bool_list) {
    # Train, valid, test
    train_woe[, v] <- as.numeric(train_woe[, v])
    valid_woe[, v] <- as.numeric(valid_woe[, v])
    
    
    # rejects
    rejects_woe[, v] <- as.numeric(rejects_woe[, v])
}

#Check infinite values and remove
#Train, valid
sum(apply(sapply(train_woe, is.infinite), 2, sum))
sum(apply(sapply(valid_woe, is.infinite), 2, sum))

# Rejects
sum(apply(sapply(rejects_woe, is.infinite), 2, sum))

# Impute +/-Inf value by NA
# Train, valid
train_woe[sapply(train_woe, is.infinite)] <- NA
valid_woe[sapply(valid_woe, is.infinite)] <- NA

# Rejects
rejects_woe[sapply(rejects_woe, is.infinite)] <- NA

# Check missing value
# Train, valid
sum(apply(is.na(train_woe), 2, sum))
sum(apply(is.na(valid_woe), 2, sum))

# Rejects (holdout)
sum(apply(is.na(rejects_woe), 2, sum))

# Impute missing value in numerical variable by mean
for (v in iv_num_list) {
    # Train, valid, test
    train_woe[is.na(train_woe[, v]), v] <- mean(train_woe[, v], na.rm=T)
    valid_woe[is.na(valid_woe[, v]), v] <- mean(valid_woe[, v], na.rm=T)
    
    
    # Rejects
    rejects_woe[is.na(rejects_woe[, v]), v] <- mean(rejects_woe[, v], na.rm=T)
}

# Check if train and test (holdout) have same variables
# Train, valid, test
dim(train_woe)
dim(valid_woe)

# Rejects (holdout) This would have one less column as final creditability is not available 
dim(rejects_woe) 


# For variable selection information value is used. This is executed via the Scorecard package
#ls("package:scorecard")
#?var_filter

iv(train_woe,"Bad_creditability",order=TRUE)

best_iv_var = var_filter(train_woe, y = "Bad_creditability",var_rm =c("woe.ID.binned","ID"))
dim(best_fs_var)

best_iv_var[["Bad_creditability"]] <- NULL

best_iv_var

# Apply variable selection to the data
# Train
var_select <- names(train_woe)[names(train_woe) %in% names(best_iv_var)]

train_processed <- train_woe[, c('ID', var_select,'Bad_creditability')]
train_processed$Bad_creditability<-as.factor(train_processed$Bad_creditability)

colnames(train_processed)

# Valid
var_select <- names(valid_woe)[names(valid_woe) %in% names(best_iv_var)]
valid_processed <- valid_woe[, c('ID', var_select,'Bad_creditability')]
valid_processed$Bad_creditability<-as.factor(valid_processed$Bad_creditability)

# Reject
var_select <- names(rejects_woe)[names(rejects_woe) %in% names(best_iv_var)]
rejects_processed <- rejects_woe[, c('ID', var_select)]
colnames(rejects_processed)

# Final check if train and test (holdout) have same variables
# Train, valid, test
dim(train_processed)
dim(valid_processed)

# Test (holdout)
dim(rejects_processed)

#####################################################

######### FIT GRADIENT BOOSTED TREE #################

#####################################################

# Fit Gradient Boosted Tree Model

## * Note : I had to add the package name mlr:: infront of functions due to functions being conflicted with other packages


#load GBM
getParamSet("classif.gbm")

#set 5 fold cross validation
rdesc <- mlr::makeResampleDesc("CV",iters = 5)

# Define model - make a learner
learner <- mlr::makeLearner("classif.gbm", predict.type="prob", fix.factors.prediction=T)

# Define the task
train_task <- mlr::makeClassifTask(id="bank_train", data=train_processed[, -1], target="Bad_creditability")


# Set hyper parameter tuning
tune_params <-mlr::makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), #number of trees
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), #depth of tree
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
ctrl = makeTuneControlGrid()

# Run the hyper parameter tuning with k-fold CV
if (length(tune_params$pars) > 0) {
  # Run parameter tuning
  res <- mlr::tuneParams(learner, task=train_task, resampling=rdesc,
                    par.set=tune_params, control=ctrl, measures=list(mlr::auc))
  
  # Extract best model
  best_learner <- res$learner
  
} else {
  # Simple cross-validation
  res <- resample(learner, train_task, rdesc, measures=list(mlr::auc))
  
  # No parameter for tuning, only 1 best learner
  best_learner <- learner
}

# Retrain the model with tbe best hyper-parameters
best_md <- mlr::train(best_learner, train_task)

# Make prediction on valid data
pred <- predict(best_md, newdata=valid_processed[, -1])

#Check the accuracy

mlr::performance(pred, measures = list(auc, acc, fpr, fnr, f1))

##The accuracy of the model is good 
#however it takes a lot of time 
#and it would take more time when trying to predict to a larger dataset

#####################################################

############ FIT LOGISTIC REGRESSION #############

#####################################################
# Next, I tried out a fairly simple however an effective model 'Logistic Regression', widely used in the industry.

# Set up cross-validation
# Cross validation is essential in order to make sure we do not overfit the data
rdesc = mlr::makeResampleDesc("CV", iters=5, predict="both")

# Define the model
learner <- mlr::makeLearner("classif.logreg", predict.type="prob", fix.factors.prediction=T)

# Define the task
train_task <- mlr::makeClassifTask(id="bank_train", data=train_processed[, -1], target="Bad_creditability")

# Set hyper parameter tuning
tune_params <- mlr::makeParamSet(
)
ctrl = mlr::makeTuneControlGrid()

# Run the hyper parameter tuning with k-fold CV
if (length(tune_params$pars) > 0) {
    # Run parameter tuning
    res <- tuneParams(learner, task=train_task, resampling=rdesc,
      par.set=tune_params, control=ctrl, measures=list(mlr::auc))
    
    # Extract best model
    best_learner <- res$learner
    
} else {
    # Simple cross-validation
    res <- mlr::resample(learner, train_task, rdesc, measures=list(mlr::auc, setAggregation(mlr::auc, train.mean)))
    
    # No parameter for tuning, only 1 best learner
    best_learner <- learner
}

#train model with best parameters
best_md <- mlr::train(best_learner, train_task)

# Make prediction on valid data
pred <- predict(best_md, newdata=valid_processed[, -1])

#Check accuracy
#performance(pred, measures=mlr::auc,acc,fpr, fnr, f1)

mlr::performance(pred, measures = list(mlr::auc, mlr::acc, mlr::fpr, mlr::fnr, mlr::f1))

# The accuracy, acc and F1 values of the model is good although fpr and fnr is a slightly high
# However I pick the ' Logistic Regression' model due to ease of use and because it is less time consuming
#  Because, the time factor would be important in a real business setting
# The GBM takes too much time though it may give better results, hence there is less value addition
# as time equals costs. 

#Change type to numeric of woe.Business.binned as there was a difference when compared with training
rejects_processed$woe.Business_channel.binned<-as.numeric(rejects_processed$woe.Business_channel.binned)


# Make prediction on reject data
rejects_pred <- predict(best_md, newdata=rejects_processed[, -1])

#In rejects data we consider .10 above are bad in terms of creditability while below may be good creditors
# The reason why I considered the prob.1 rates was that I thought it would help the performance of the model better
# for the reject inference to consider the probabilities derived from the model which used accepts dataset
# However, it may be biased as the model was created using only the accepts data set.

predicted.bad <- as.data.frame(ifelse(rejects_pred$data[['prob.1']]> 0.10, 1, 0))
head(predicted.bad, 10)

#Assign the same name for the response
names(predicted.bad)[1] <- "Bad_creditability"

#Checking if the number of rows match the processed rejects data set
dim(predicted.bad)

#Assigning type factor for bad creditability
predicted.bad$Bad_creditability<-as.factor(predicted.bad$Bad_creditability)

#Checking if the bad_creditability rate is close to 75% of the dataset
length(predicted.bad$Bad_creditability[predicted.bad$Bad_creditability==1])


# Binding the values based on the probabilities with the rejects data set
rejects_processed_pred<-cbind(rejects_processed,predicted.bad)

head(rejects_processed_pred)

######## Preparation of final training data ########


# Check if dimensions of both datasets are the same
# This makes sure that both sets contains the same variables which is important when binding
dim(rejects_processed_pred)
dim(train_processed)

# Bind the two data set to create one large training set
total_train<-rbind(train_processed,rejects_processed_pred)

dim(total_train)

############ FIT LOGISTIC REGRESSION FOR BOTH DATASETS #############

#Next we train the model with data set created combining both accepts and rejects data set as this would provide a more realistic prediction,
#when predicting the creditability of the customers

# Set up cross-validation
rdesc = mlr::makeResampleDesc("CV", iters=5, predict="both")

# Define the model
learner <- mlr::makeLearner("classif.logreg", predict.type="prob", fix.factors.prediction=T)

# Define the task
train_task <- mlr::makeClassifTask(id="bank_train", data=total_train[, -1], target="Bad_creditability")

# Set hyper parameter tuning
tune_params <- mlr::makeParamSet(
)
ctrl = mlr::makeTuneControlGrid()

# Run the hyper parameter tuning with k-fold CV
if (length(tune_params$pars) > 0) {
  # Run parameter tuning
  res <- tuneParams(learner, task=train_task, resampling=rdesc,
                    par.set=tune_params, control=ctrl, measures=list(mlr::auc))
  
  # Extract best model
  best_learner <- res$learner
  
} else {
  # Simple cross-validation
  res <- resample(learner, train_task, rdesc, measures=list(mlr::auc, setAggregation(mlr::auc, train.mean)))
  
  # No parameter for tuning, only 1 best learner
  best_learner <- learner
}

#train model with best parameters
best_md <- mlr::train(best_learner, train_task)

# Make prediction on valid data
pred_new <- predict(best_md, newdata=valid_processed[, -1])

#Check the predictions
pred_new

#Check accuracy
#Because of conficts with other packages I had to mention mlr:: for each accuracy measure

mlr::performance(pred_new, measures = list(mlr::auc, mlr::acc, mlr::fpr, mlr::fnr, mlr::f1))

#The auc is 0.753 and has declined slightly with the new model but only by 0.1 however false positive
#and false negative rate have drastically declined 
# In other words, this means that falsely rejecting a null hypothesis has gone down
#False positive rate means the number of negative items which are wrongly classified as positive in this case as individuals with 'Bad creditability' but are actually 'Good'.
# while false negative means number which is falsely classified as positive however when it is not, in this case individuals who are marked as 'Good' but are actually bad.
# False positive rate also provides the significance of the result. (Significance = 1-FPR) Hence you can see that significance is good and has improved with the latest results.


#############################################################
##############################################################
################## Survival Analysis #########################
##############################################################

 install.packages("survival")
# Loading the package
library("survival")

#Read in the two data files
credit_s <- read.table("creditsurv.csv", header = T) 
head(credit_s)

# Investigate the variables 
summary(credit_s)

#Identify time and status variable to incorporate survival analysis
#Curradd: years at address
#Curremp: years at current work place
#Homephon:  Has home phone or not
#Marstat : Marital status
#Homeowns:  Has home or not

#Creating survival function for Censore - assuming it's early payment

surv=survfit(Surv(credit_s$Curradd,credit_s$Curremp,credit_s$Homephon,credit_s$Marstat,credit_s$Homeowns,credit_s$Censore == 1)~1)
surv

#Plot the model to view the probabilities for survival as time passes

plot(surv)

