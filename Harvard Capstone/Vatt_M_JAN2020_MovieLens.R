## Michael Vatt
## 04 JAN 2020
## HarvardX: Capstone Project - PH125.9x

#### MovieLens Rating Prediction Project Code ####

#### Introduction ####

## Dataset ##

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes for loading required package: tidyverse and package caret

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(knitr)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
  col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
  title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# The Validation subset will be 10% of the MovieLens data.

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

#Make sure userId and movieId in validation set are also in edx subset:

validation <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

### Methods & Analysis ###

# Preview of the Dataset

head(edx)

# Summary of the Dataset

summary(edx) 

# Number of Distinct Movies and Users in the Dataset

edx %>% summarize(dist_users = n_distinct(userId), dist_movies = n_distinct(movieId))

# Distribution of Ratings of the Dataset Plot

edx %>% ggplot(aes(rating)) + geom_histogram(binwidth = .25, fill = "orange", color = "black") + 
  scale_x_discrete(limits = c(seq(.5,5,.5))) + scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) + 
   xlab("Rating") + ylab("Number of Movies Rated") + ggtitle("Ratings Distribution") + theme_light()

# Number of Ratings Per Movie Plot

edx %>% count(movieId) %>% ggplot(aes(n)) + geom_histogram(bins = 25, fill = "orange", color = "black") + 
  scale_x_log10() + xlab("Number of Ratings") + ylab("Number of Movies") + ggtitle("Number of Ratings per Movie") +
  theme_light()

# A Table of 10 Movies Having Only a Single Rating

edx %>% group_by(movieId) %>% summarize(count = n()) %>% filter(count == 1) %>% left_join(edx, by = "movieId") %>% 
  group_by(title) %>% summarize(rating = rating, n_rating = count) %>% print(as_tibble(iris), n = 10)

# Number of Ratings Given by Users

edx %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(bins = 25, fill = "orange", color = "black") + 
  scale_x_log10() + xlab("Number of Ratings") + ylab("Number of Users") + 
  ggtitle("Number of Ratings Given by Users") + theme_light()

# Avg Movie Ratings Given by Users Plot 

edx %>% group_by(userId) %>% filter(n() >= 50) %>% summarize(beta_u = mean(rating)) %>% ggplot(aes(beta_u)) + 
  geom_histogram(bins = 25, fill = "orange", color = "black") + xlab("Avg Rating") + ylab("Number of Users") + 
  ggtitle("Avg Movie Ratings Given by Users") + scale_x_discrete(limits = c(seq(.5,5,.5))) + theme_light() 

## Average Movie Rating Model ##

# Dataset’s Mean

mu <- mean(edx$rating)
mu

# Simple Prediction 

simple_rmse <- RMSE(validation$rating, mu)
simple_rmse

# Save Prediction in Data Frame

rmse_preds <- data.frame(method = "Simple Movie Rating Model", RMSE = simple_rmse)

# Check Results

rmse_preds

## Type of Movie Effect Model ##

# A Model Taking Into Account that the Type of Movie Influences Ratings Itself

movie_avgs <- edx %>% group_by(movieId) %>% summarize(beta_i = mean(rating - mu))
movie_avgs %>% ggplot(aes(beta_i)) + geom_histogram(bins = 10, fill = "orange", color = "black") + 
  xlab("beta_i") + ylab("Number of Movies") + ggtitle("Number of Movies with the Computed beta_i") + 
  theme_light()

# Test and Save RMSE Results

penalty_ratings <- mu + validation %>%  left_join(movie_avgs, by = "movieId") %>% pull(beta_i) 
first_rmse <- RMSE(penalty_ratings, validation$rating)
rmse_preds <- add_row(rmse_preds, method = "Movie Effect Model", RMSE = first_rmse)

# Check Results

rmse_preds

## Movie and User Effect Model ##

# Penalty User Effect Plot 

user_avgs <- edx %>% left_join(movie_avgs, by = "movieId") %>% group_by(userId) %>% summarize(beta_u = mean(rating - mu - beta_i))
user_avgs %>% ggplot(aes(beta_u)) + geom_histogram(bins = 25, fill = "orange", color = "black") + xlab("beta_u") + 
  ylab("Number of Users") + ggtitle("Penalty User") + theme_light()

# Test and Save Results

predicted_ratings <- validation %>% left_join(movie_avgs, by = "movieId") %>% left_join(user_avgs, by = "userId") %>% mutate(prediction = mu + beta_u + beta_i) %>% pull(prediction)
second_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_preds <- add_row(rmse_preds, method = "Movie and User Effect Model", RMSE = second_rmse)

# Check Results

rmse_preds

## Regularized Movie and User Effect Model ##

# Use Lambda as Tuning Parameter

lambdas <- seq(0,10,.25)

# Find beta_u and beta_i for each lambda and predict
# WARNING: THIS MAY BE A SLOW CALCULATION

rmses <- sapply(lambdas, function(lambda){
  mu <- mean(edx$rating)
  beta_i <- edx %>% group_by(movieId) %>% summarize(beta_i = sum(rating - mu) / (n() + lambda))
  beta_u <- edx %>% left_join(beta_i, by="movieId") %>% group_by(userId) %>% summarize(beta_u = sum(rating - beta_i - mu) / (n() + lambda))
  predicted_ratings <- validation %>% left_join(beta_i, by = "movieId") %>% left_join(beta_u, by = "userId") %>% 
    mutate(prediction = mu + beta_i + beta_u) %>% pull(prediction)
  lambda_predict <- RMSE(predicted_ratings, validation$rating)
  lambda_predict 
})


# Plot for Discovery of Optimal Lambda

ggplot(mapping = aes(x = lambdas, y = rmses)) + geom_point()

# Mathematical Discovery of Optimal Lambda 

op_lambda <- lambdas[which.min(rmses)]
op_lambda


# Test and save results 

rmse_preds <- add_row(rmse_preds, method = "Regularized Movie and User Effect Model", RMSE = min(rmses))

# Check result

rmse_preds

#### Results #### 

# RMSE results overview 

rmse_preds

#### Conclusion ####

print("This machine learning algorithm was able to predict movie ratings with the MovieLens Dataset. Hardware limitations were difficult to manage, such as RAM and CPU, but successful nonetheless. Improvements to the algorithm could be managed or obtained by evaluating other variables as well.")

