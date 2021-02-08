## Michael Vatt
## 16 JUN 2020
## OHR4

#### Employee Turnover Code ####

# Load the following packages

# Note: This Process Could Take a Couple of Minutes for Loading Required Packages

install.packages("pacman", repos = "http://cran.us.r-project.org")

pacman::p_load("anytime", "bit", "car", "caret", "caTools", "data.table", "doParallel", 
               "dplyr", "e1071", "ggplot2", "ggpubr", "glmnet", "gridExtra", "h2o", "jsonlite", 
               "knitr", "lava", "lime", "lubridate", "methods", "pdp", "RColorBrewer", "RCurl", 
               "readr", "readxl", "rjson", "scales", "statmod", "scales", "stats", "stringi",
               "stringr", "survival", "tibble", "tidyquant", "tidyr", "tidyverse", "timeDate", 
               "tinytex", "tools", "utils", "versions", "vip")

# Recent Changes to h2o package so we will perform the following as a precaution

# The following two commands remove any previously installed H2O packages for R

if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on

pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R

 install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-zahradnik/5/R")


 # Finally, let's load H2O and start up an H2O cluster

 library(h2o)

# Read excel data

# Fake dataset created by IBM Watson
hr_data_raw <- read_excel(path = "C:/My Desktop/OHR/Projects/WA_Fn-UseC_-HR-Employee-Attrition.xlsx")

class(hr_data_raw)
# View first 10 rows
#hr_data_raw[1:10,] %>%
#  knitr::kable(caption = "First 10 rows")  # This version looks great in pdf/html with knitr, but not in the console
hr_data_raw[1:10,]

# Pre-processing change character data types to factors. This is needed for H2O

# The Attrition column is our target - we'll use all other Cols as features.
hr_data <- hr_data_raw %>% mutate_if(is.character, as.factor) %>% 
  select(Attrition, everything()) # everything() selects all variables

# 1470 rows and 35 cols (features)
glimpse(hr_data)

# First we need to initialize the Java Virtual Machine (JVM) that H2O uses locally

h2o.init()

# Turn off output of progress bars

h2o.no_progress() 

# Split the data into Train/Validation/Test Sets
  # This is one sort of method (many have simply Train/Test)
  # Adding a Validation Set enables an estimation of the model skill while tuning model's hyperparameters
  # Used to give an unbiased estimate of the final tuned model 
  # There are other methods to calculate an unbiased estimate (e.g., k-fold cross-validation)

hr_data_h2o <- as.h2o(hr_data)

split_h2o <- h2o.splitFrame(hr_data_h2o, c(0.7, 0.15), seed = 1234)

train_h2o <- h2o.assign(split_h2o[[1]], "train" ) # 70%
valid_h2o <- h2o.assign(split_h2o[[2]], "valid" ) # 15%
test_h2o  <- h2o.assign(split_h2o[[3]], "test" )  # 15%

index <- sample(seq(1,3), size = nrow(hr_data), replace = TRUE, prob = c(.7, .15, .15))

train_set <- hr_data[index == 1,]
valid_set <- hr_data[index == 2,]
test_set <- hr_data[index == 3,]

# Model - we aim to predict Attrition and the features (other Cols) are used to model the prediction

# Set names for h2o

y <- "Attrition"
x <- setdiff(names(train_h2o), y)

# Run the automated machine learning 

# x = x: names of our feature cols
# y = y: name of our target col
# training_frame = train_h2o: training set of 70% of the data
# leaderboard_frame = valid_h2o: validation set of 15% of the data
  # this is to ensure the model does not overfit the data
# max_runtime_secs = 30: this is to speed up h2o's modeling
  # algorithm has number of large complex models - this is expedition at the expense of some accuracy

automl_models_h2o <- h2o.automl(
  x = x, 
  y = y,
  training_frame    = train_h2o,
  leaderboard_frame = valid_h2o,
  max_runtime_secs  = 30)

# h2o.xgboost.available() (currently only for Mac and Linux OS)

# All of the models are stored in the automl_models_h2o object
# However, we only care about the leader, which is the best model in terms of accuracy on the vaildation set

# View the AutoML Leaderboard

lb <- automl_models_h2o@leaderboard
print(lb, n = nrow(lb)) # Print all rows instead of default 6 rows

# Extract leader model

automl_leader <- automl_models_h2o@leader
automl_leader

# Now we can predict on our test set, which is unseen from our modeling process
# Meaning, it is a true test of performance

# Predict on hold-out set, test_h2o

pred_h2o <- h2o.predict(object = automl_leader, newdata = test_h2o)

# Evaluate our model - we'll reformat the test set to add predictions col to analyze actual vs. predictions side-by-side
# Prep for performance assessment

test_performance <- test_h2o %>%
  tibble::as_tibble() %>%
  select(Attrition) %>%
  add_column(pred = as.vector(pred_h2o$predict)) %>%
  mutate_if(is.character, as.factor)
test_performance

# We can use the table() function to quickly get a confusion table of the results. 
# We see that the leader model wasn't perfect, but it did a decent job at identifying
# employees that are likely to quit. 
# For perspective, a logistic regression would not perform nearly this well

# Confusion table counts

confusion_matrix <- test_performance %>% table() 
confusion_matrix

# We can run through a binary classification analysis to understand the model performance

# Performance analysis

tn <- confusion_matrix[1] # true negatives
tn
tp <- confusion_matrix[4] # true positives
tp
fp <- confusion_matrix[3] # false positives
fp
fn <- confusion_matrix[2] # false negatives
fn

accuracy <- (tp + tn) / (tp + tn + fp + fn)
misclassification_rate <- 1 - accuracy
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
null_error_rate <- tn / (tp + tn + fp + fn)

tibble(
  accuracy, 
  misclassification_rate, 
  recall, 
  precision, 
  null_error_rate
) %>% 
  transpose() 

perf <- data.frame(Method = "Accuracy", Percentage = percent(accuracy))
perf <- add_row(perf, Method = "Misclassification Rate", Percentage = percent(misclassification_rate))
perf <- add_row(perf, Method = "Recall", Percentage = percent(recall))
perf <- add_row(perf, Method = "Precision", Percentage = percent(precision))
perf <- add_row(perf, Method = "Null Error Rate", Percentage = percent(null_error_rate))

#install.packages("AUC")
#library(AUC)

cm <- confusionMatrix(data = test_performance$pred, reference = test_performance$Attrition)

#sensitivity(test_performance$pred, test_performance$Attrition)
#specificity(test_performance$pred, test_performance$Attrition)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'No', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Yes', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'No', cex=1.2, srt=90)
  text(140, 335, 'Yes', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
  
  
}  

#names(cm)

draw_confusion_matrix(cm)

#head(cm)

#cm$byClass






# Changed to percentages since that is usually easier for people to read

perf

# It is important to understand that accuracy can be misleading
# 88% accuracy sounds pretty good - especially for modeling HR data
# But if we simply just picked Attrition = No, we would get an accuracy of about 78%
# Doesn't sound so great now.

# Precision is when the model predicts yes, how often it is actually yes
# Recall (aka true positive rate or specificity) is when the actual value is yes, how often the model is correct
# Most HR divisions would rather incorrectly classify folks not looking to quit as high potential than 
# classify those likely to quit as not at risk. HR will then care about Recall
# Recall - when the actual value is Attrition = YES, how often the model predicts YES.
# Recall for our model is 69% - in an HR context, this is 69% more employees that could potentially
# be targeted prior to quitting. Let's say an organization loses 100 employees per year,
# they could possibly target 69 of them, implementing measures to retain.

# Thus far, we have a very good model that is capable of making very accurate predictions on unseen data,
# but what can it tell us about what causes attrition? Here, we can use LIME. 

# The lime package implements LIME in R. 
# NOTE: lime is not setup out-of-the-box to work with h2o
# Two custom functions will enable everything to work smoothly:

# model_type: Tells lime what type of model we are dealing with
# predict_model: Allows lime to perform predictions that its algorithm can interpret

class(automl_leader)

# Build Model Type function for congruence with h2o

model_type.H2OBinomialModel <- function(x, ...) {
  
  # Function tells lime() what model type we are dealing with
  # 'classification', 'regression', 'survival', 'clustering', 'multilabel', etc
  #
  # x is our h2o model
  
  return("classification")
}

# Build predict_model function

# Trick here is to realize that it's inputs must be 'x' (a model), 'newdata' (a dataframe object - this is essential),
# and 'type' (not used, but can be used to switch the output type).
# Output is also tricky because it MUST be in the format of probabilities by classification

predict_model.H2OBinomialModel <- function(x, newdata, type, ...) {
  
  # Function performs prediction and returns dataframe with Response
  #
  # x is h2o model
  # newdata is data frame
  # type is only setup for data frame
  
  pred <- h2o.predict(x, as.h2o(newdata))
  
  # return probs
  return(as.data.frame(pred[,-1]))
  
}

# We can run the next script to show what the output looks like and test our predict_model function

# Test our predict_model() function

predict_model(x = automl_leader, newdata = as.data.frame(test_h2o[,-1]), type = 'raw') %>%
  tibble::as_tibble()

# Now the fun part, we create an explainer using the lime() function
# Pass the training dataset without the "Attribution column". Form must be a data frame.
# Our predict_model function will switch it to an h2o object
# Set model = autol_leader and bin_continuous = FALSE.
# We could do bin_continuous variables, but this may not make sense for categorical numeric
# data that we didn't change to factors.

# Run lime() on training set

explainer <- lime::lime(as.data.frame(train_h2o[,-1]), model = automl_leader, 
                        bin_continuous = FALSE)

# Now run explain(), which returns our explanation. This can take a few minutes to run, so let's just limit
# it to the first 10 rows of the test dataset. 

# Set n_labels = 1 because we care about explaining a single class. 
# Setting n_features = 4 returns the top four features that are critical to each case. 
# Setting kernel_width = 0.5 allows us to increase the "model_r2" value by shrinking the localized evaluation

# Run explain() on explainer

explanation <- lime::explain(
  as.data.frame(test_h2o[1:10,-1]), 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 6,
  kernel_width = 0.5)

# Feature Importance Visualization

# The payoff for all this work is the Feature Importance Plot.
# We can visualize each of the ten cases (observations) from the test dataset.
# The top four features for each case are shown.NOTE: they are not the same for each case.
# Green bars mean that the feature supports the model conclusion, red bars contradict.

# Focus on cases with Lable = YES - which are predicted to have attrition. 
# Are there common themes?
# They may only exist in a couple cases, but they can be used to potentially generalize to the larger population.

plot_features(explanation) +
  labs(title = "HR Predictive Analytics: LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

# What features are linked to Employee Attrition?
  
  # Training Time
  # Job Role
  # Overtime

# Focus on critical features of attrition

attrition_critical_features <- hr_data %>% tibble::as_tibble() %>%
  select(Attrition, TrainingTimesLastYear, JobRole, OverTime) %>%
  rowid_to_column(var = "Case")
attrition_critical_features

# Training

violin plot

# Job Role

horizontal bar graph

# Overtime

violin plot
  
# Conclusions

# The autoML algorithm from H2O.ai worked well for classifying attrition with an accuracy around 88% 
# on unseen / unmodeled data. 

# We then used LIME to breakdown the complex ensemble model returned from H2O into critical features that are related to attrition. 

# OVerall, this is a really useful example where we can see how much ML and DS can be used in HR applications. 

