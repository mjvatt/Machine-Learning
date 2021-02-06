# 2) Multivariate Linear Regression House Price Prediction

# Install/Load Necessary Packages
  # packages commented out if already installed
  # Algorithm to install packages left in the script in case it was needed. 


# Create a Function to Check for Installed Packages and Install if They Are Not Installed

install <- function(packages){
  new.packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new.packages))
    install.packages(new.packages, dependencies = TRUE, repos = "http://cran.us.r-project.org")
  sapply(packages, require, character.only = TRUE)
}

# Install

packages <- c("caTools", "car", "caret", "cluster", "Clustering", "corpus", "corrplot", "data.table",
              "dendextend", "doParallel", "dplyr", "e1071", "factoextra", "FactoMineR", "fpc",
              "GGally", "ggplot2", "ggthemes", "gridExtra", "kableExtra", "knitr", "ldatuning",
              "magrittr", "mclust", "NbClust", "petro.One", "plotly", "plotrix", "qdap", "qdapTools",
              "quanteda", "randomForest", "readxl", "RColorBrewer", "rlist", "RWeka", "scales",
              "SentimentAnalysis", "sentimentr", "SnowballC", "stm", "stringr", "syuzhet",
              "tensorflow", "tidyr", "tidytext", "tidyverse", "tm", "topicmodels", "viridisLite",
              "wordcloud", "xlsx", "zoo")

install(packages)



# Call to load the installed packages also commented out but left in the script in case it was
  # needed to perform the script. 



# Call the installed packages

#library(plyr) # plyr is required to be loaded before dplyr or issues may arise
library(dplyr) # dplyr needed for efficient loading of loadApp()

loadApp <- function() {

  my_library <- c("caTools", "car", "caret", "cluster", "Clustering", "corpus", "corrplot", "data.table",
                  "dendextend", "doParallel", "dplyr", "e1071", "factoextra", "FactoMineR", "fpc",
                  "GGally", "ggplot2", "ggthemes", "gridExtra", "kableExtra", "knitr", "ldatuning",
                  "magrittr", "mclust", "NbClust", "petro.One", "plotly", "plotrix", "qdap", "qdapTools",
                  "quanteda", "randomForest", "readxl", "RColorBrewer", "rlist", "RWeka", "scales",
                  "SentimentAnalysis", "sentimentr", "SnowballC", "stm", "stringr", "syuzhet",
                  "tensorflow", "tidyr", "tidytext", "tidyverse", "tm", "topicmodels", "viridisLite",
                  "wordcloud", "xlsx", "zoo")

  install.lib <- my_library[!my_library %>% installed.packages()]

  for(lib in install.lib) install.packages(lib, dependencies = TRUE)

  sapply(my_library, require, character = TRUE)

}

loadApp()


# Import Data set

dataset <- read.csv("https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv")

# Initial Familiarization with the Data set and to Ensure it was Properly Read In

class(dataset) # Need to coerce into data frame if it is not
dim(dataset) #  For familiarity and a quick visualization to determine if something went wrong
summary(dataset) # Does anything look strange? Any variable(s) requiring pre-processing?
str(dataset)  # There are 6 variables of numerical and 1 variable factor
head(dataset) # Numerical variables need scaling? Should we not include any variables?
colnames(dataset) <- c("Income", "House_Age", "Number_Rooms", "Number_Bedrooms", "Area_Population", 
                       "Price", "Address") # For easier handling and better visualization
names(dataset) # Ensure that the variable names changed

## Will Limit Regression to the First 6 Variable Columns and Leave out Address for the Predictions
## Address would require encoding. Since it would create 1 dummy variable per observation (5000),
## encoding would be trivial and inconsequential for a prediction. 

# Let's Check for Missing Data

data.frame(count_na_values <- colSums(is.na(dataset))) # Need to appraise if any greater than 0.

# Split the Data set into a Training Set and Test Set

set.seed(1234)
val_ind <- createDataPartition(dataset$Price, times = 1, p = 0.2, list = FALSE) # Split 80/20
train_set <- dataset[-val_ind,] # 4000 observations for training data
test_set <- dataset[val_ind,] # 1000 observations to test upon

# Let's Make Sure the Split Operated Correctly

dim(train_set); dim(test_set) # 80% and 20% of the data set respectively
class(train_set); class(test_set) # Should be data.frames

# Remove Address From Data set for Prediction

train_set <- train_set[,(1:6)] # Only keep numerical variables significant to the regression prediction
test_set <- test_set[,(1:6)] # Only keep numerical variables significant to the regression prediction
names(train_set); names(test_set) # Ensure proper variables were kept

## Model Will Not Need to Perform Feature Scaling - Linear Regression Function lm() Handles that Inherently

# Fitting Multiple Linear Regression to the Training Set

model1 <- lm(formula = Price ~ ., data = train_set) # Price is the dependent variable as a function of all dependent variables via '.'
summary(model1) 

## We see that Number_Bedrooms is not statistically significant - would removing this variable aid prediction accuracy?

# Using Fitted model1 for Predicting the Test Set Results

y_pred1 <- predict(model1, newdata = test_set)
visual_comparison <- data.frame(test_set$Price, y_pred, residuals = test_set$Price - y_pred1)
visual_comparison[1:10,]

## Despite the model returning statistically significant p-value and R-squared values, we can see 
## the model did not fit quite well on the test data set. Many of the predicted values have 
## a noticeable difference. It is possible to make the model more robust and vigorous through more 
## advanced models. Checking for multicollinearity may also help improve the algorithm.

# Let's try to Evaluate Associations between Variables

correlation <- cor(train_set)
corrplot(correlation, method = "color")

## Price is positively correlated with Income, House_Age, Number_Rooms, and Area_Population
## Price has a slight positive correlation with Number_Bedrooms

# Let's Make Scatter Plots to Determine the Relationship(s) between the Variables

bedroom_plot <- ggplot(data = train_set, aes(x = Number_Bedrooms, y = Price)) + geom_jitter() +  
  geom_smooth(method = "lm", se = FALSE) + 
  labs(title = "Scatter plot of Bedrooms and Price", x = "Number of Bedrooms", y = "Price")

bedroom_plot # We see that the relationship is linear

income_plot <- ggplot(data = train_set, aes(x = Income, y = Price)) + geom_jitter() +  
  geom_smooth(method = "lm", se = FALSE) + 
  labs(title = "Scatter plot of Income and Price", x = "Income", y = "Price")

income_plot # We see that the relationship is linear

age_plot <- ggplot(data = train_set, aes(x = House_Age, y = Price)) + geom_jitter() +  
  geom_smooth(method = "lm", se = FALSE) + 
  labs(title = "Scatter plot of House Age and Price", x = "Age of House", y = "Price")

age_plot # We see that the relationship is linear

room_plot <- ggplot(data = train_set, aes(x = Number_Rooms, y = Price)) + geom_jitter() +  
  geom_smooth(method = "lm", se = FALSE) + 
  labs(title = "Scatter plot of Rooms and Price", x = "Number of Rooms", y = "Price")

room_plot # We see that the relationship is linear

pop_plot <- ggplot(data = train_set, aes(x = Area_Population, y = Price)) + geom_jitter() +  
  geom_smooth(method = "lm", se = FALSE) + 
  labs(title = "Scatter plot of Area Population and Price", x = "Area Population", y = "Price")

pop_plot # We see that the relationship is linear

# Evaluate Correlations Amongst Variables

ggpairs(train_set, columns = c(1:6))

## Correlation evaluation appears to support that Number_Bedrooms has a much lower correlation
## and statistical significance towards Price. 

# Check for Outliers

ggplot(data = train_set) + geom_boxplot(aes(x = Number_Bedrooms, y = Price)) # Some outliers exist
train_set_without_outliers <- train_set[-which(train_set$Price %in% boxplot(train_set$Price, plot = FALSE)$out),]

## There were 34 outliers, uncertain if their removal will significantly impact overall prediction

# Visualize Outlier Differences

rounded_without_outliers <- round(train_set_without_outliers)
rounded_with_outliers <- round(train_set)

plot1 <- ggplot(data = rounded_without_outliers, aes(x = Income, y = Price)) +
  geom_point() + geom_smooth(method = lm) + xlim(0, 200000) + ylim(0, 2000000) + 
  ggtitle("No Outliers")

plot2 <- ggplot(data = rounded_with_outliers, aes(x = Income, y = Price)) +
  geom_point() + geom_smooth(method = lm) + xlim(0, 200000) + ylim(0, 2000000) + 
  ggtitle("With Outliers")

grid.arrange(plot1, plot2, ncol = 2)

# Fitting Multiple Linear Regression to the Training set Without Number_Bedrooms

train_set_without_outliers <- train_set_without_outliers[,c(1:3,5:6)] # Remove Number_Bedrooms variable

model2 <- lm(formula = Price ~ ., data = train_set_without_outliers) # Price is the dependent variable as a function pf all dependent variables via '.'
summary(model2) 

## All independent variables are now statistically dependent. R-squared and p-values remain 
## statistically significant, however, they both actually decreased by approximately 0.005

# Using Fitted model2 for Predicting the Test Set Results

y_pred2 <- predict(model2, newdata = test_set)
visual_comparison2 <- data.frame(test_set$Price, y_pred, residuals = test_set$Price - y_pred2)
visual_comparison2[1:10,]

## Predictions were slightly different but essentially no improvement. 

# Let's Detect Influential Data Points with Cook's Distance
# These are more influential than simple outliers

cooksd <- cooks.distance(model2)
mean(cooksd)

plot(cooksd, main = "Influential Observations by Cook's Distance", xlim = c(0,5000), ylim = c(0,0.005))
abline(h = 4 * mean(cooksd, na.rm=T), col = "green")  # Creates cutoff to find influential points
text(x = 1:length(cooksd) + 1, y = cooksd, labels = ifelse(cooksd > 4 * mean(cooksd, na.rm = T), 
                                                           names(cooksd), ""), col = "red")  

influential <- as.numeric(names(cooksd)[(cooksd > 4 * mean(cooksd, na.rm=T))])  # influential row numbers
head(train_set_without_outliers[influential, ]) # Take a peek at a few influntial observations

train_set_without_outliers_influential <- train_set_without_outliers[-influential, ]
nrow(train_set_without_outliers_influential)

plot3 <- ggplot(data = rounded_without_outliers, aes(x = Income, y = Price)) + geom_point() + 
  geom_smooth(method = lm) + xlim(0, 200000) + ylim(0, 2000000) +  ggtitle("Without Outliers")

plot4 <- ggplot(data = train_set_without_outliers_influential, aes(x = Income, y = Price)) +
  geom_point() + geom_smooth(method = lm) + xlim(0, 200000) + ylim(0, 2000000) + 
  ggtitle("Without Outliers + Influential")

grid.arrange(plot3, plot4, ncol = 2)

# Fitting Multiple Linear Regression to the train_set3

model3 <- lm(formula = Price ~ ., data = train_set_without_outliers_influential) # Price is the dependent variable as a function pf all dependent variables via '.'
summary(model3) 

## Initial review, model3 virtually fit the same as model2

# Predicting the Test Set Results on train_set3

y_pred3 <- predict(model3, newdata = test_set)
visual_comparison3 <- data.frame(test_set$Price, y_pred, residuals = test_set$Price - y_pred3)
visual_comparison3[1:10,]

## Predictions were slightly different but essentially no improvement. 
## Likely due to the fact that we had 4000 observations, outliers and influential only accounted 
## for 193 total observations. (model1 = 4000; model2 = 3966; model3 = 3807)

# Accuracy of the Models on the Different Training Sets

pred1 <- model1$fitted.values
pred2 <- model2$fitted.values
pred3 <- model3$fitted.values

tally_table <- data.frame(actual = train_set$Price, predicted = pred1)
mape <- mean(abs(tally_table$actual - tally_table$predicted) / tally_table$actual)
accuracy1 <- 1 - mape

tally_table <- data.frame(actual = train_set_without_outliers$Price, predicted = pred2)
mape <- mean(abs(tally_table$actual - tally_table$predicted) / tally_table$actual)
accuracy2 <- 1 - mape

tally_table <- data.frame(actual = train_set_without_outliers_influential$Price, predicted = pred3)
mape <- mean(abs(tally_table$actual - tally_table$predicted) / tally_table$actual)
accuracy3 <- 1 - mape

mape_preds <- data.frame(Method_on_Training_Set = c("Full Training Set", "Without Outliers", "Without Outliers + Influential"), 
                         MAPE = percent(c(accuracy1, accuracy2, accuracy3)))
mape_preds

## We see that the accuracy was actually the best on model2 for the training set

# Accuracy on test_set

tally_table_2 <- data.frame(actual = test_set$Price, predicted = y_pred1)
mape_test <- mean(abs(tally_table_2$actual - tally_table_2$predicted) / tally_table_2$actual)
accuracy_test1 <- 1 - mape_test

tally_table_2 <- data.frame(actual = test_set$Price, predicted = y_pred2)
mape_test <- mean(abs(tally_table_2$actual - tally_table_2$predicted) / tally_table_2$actual)
accuracy_test2 <- 1 - mape_test

tally_table_2 <- data.frame(actual = test_set$Price, predicted = y_pred3)
mape_test <- mean(abs(tally_table_2$actual - tally_table_2$predicted) / tally_table_2$actual)
accuracy_test3 <- 1 - mape_test


mape_preds_test <- data.frame(Method_on_Test_Set = c("Full Training Set", "Without Outliers", "Without Outliers + Influential"), 
                              MAPE = percent(c(accuracy_test1, accuracy_test2, accuracy_test3)))

mape_preds_test

## We see that the accuracy on the test set were virtually the same for each model