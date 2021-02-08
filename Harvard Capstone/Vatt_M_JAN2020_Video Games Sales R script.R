## Michael Vatt
## 09 JAN 2020
## HarvardX: Capstone Project - PH125.9x

#### Video Games Sales Prediction Project Code ####

#### Introduction ####

## Dataset ##

#############################################################
# Create Test Set, Validation Set, and Final Prediction
#############################################################


#### Video game sales from Vgchartz and corresponding ratings from Metacritic ####

# This dataset was motivated by Gregory Smith's web scrape of VGChartz Video Games Sales, this data set simply extends the number of variables with another web scrape from Metacritic. Unfortunately, there are missing observations as Metacritic only covers a subset of the platforms. Also, a game may not have all the observations of the additional variables discussed below. Complete cases are ~ 6,900

# Here are the variables that we will use in this data

# variables <- c("Name", "Platform", "Year_of_Release", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Developer")
# variables %>% kable(col.names = "Dataset Variables")

# Note: This Process Could Take a Couple of Minutes for Loading Required Packages

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

## Dataset ##

# WARNING: Some Models in this Project Require a Lot of Computing Power. Splitting the Cores Will Aid in Parallel Processing of CPUs

# NOTE: The Support Vector Machine and Random Forest Models Will Take the Longest Time

split <- detectCores()
split # To Make Sure it Worked
cl <- makePSOCKcluster(split)
registerDoParallel(cl)

# Let’s Load the Dataset from a CSV File Located in my GitHub Account

# dl <- tempfile()
# download.file("https://github.com/lkolodziejek/VideoGameSales/blob/master/Video_Games_Sales_as_at_22_Dec_2016.csv", dl)
# vgs <- fread("Video_Games_Sales_as_at_22_Dec_2016.csv", header=TRUE)

data <- read.csv("https://raw.githubusercontent.com/mjvatt/Machine-Learning/main/Harvard%20Capstone/Video_Games_Sales_as_at_22_Dec_2016.csv")
vgs <- data.frame(data)

# Familiarization with the Dataset:

dim(vgs) # For Familiarity
str(vgs) # Notice that Some Columns Are Not The Same Class
class(vgs) # Let's Ensure it is Correct Format

# Now that we are familiar with the dataset, we need to do some data wrangling to clean and prepare the data for ease of use.
 
vgs$User_Count <- as.numeric(as.character(vgs$User_Count))
vgs$User_Score <- as.numeric(as.character(vgs$User_Score))

vgs2 <- na.omit(vgs)
dim(vgs2) # To See How it Has Changed 

# First, Let's Make Some Company Variables

microsoft <- c('X360', 'XB', 'XOne')
nintendo <- c('Wii', 'NES', 'GB', 'DS', 'SNES', 'WiiU', '3DS', 'GBA', 'GC', 'N64')
sony <- c('PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV')

# Here We Define the Criteria to Identify What Belongs to Each Company

companies <- function(c) {
  if(c %in% microsoft) {return('Microsoft')}
  else if(c %in% nintendo) {return('Nintendo')}
  else if(c %in% sony) {return('Sony')}
}

# Now We Can Create This New Column

vgs2$companies <- sapply(vgs2$Platform, companies)

# Let's Check it Out to Make Sure it Worked

vgs2$companies[9:15] 


#### Methods and Analysis ####

## Exploratory Data Analysis

# Distribution of Global Sales Across Genres and Ratings

vgs2 %>% ggplot(aes(log(Global_Sales))) + stat_density(aes(color = Genre), geom = "line") + 
  labs(x = "Log Transformation of Global Sales", y = "Density") + ggtitle("Global Sales Across Genre") + 
  theme_classic()

vgs2 %>% select(Rating) %>% filter(!Rating == '') %>% count(Rating) %>% 
  ggplot(aes(Rating, n, fill = Rating, color = Rating)) + 
  geom_bar(stat = "identity") + ylab("Count of Games Per Rating") + 
  ggtitle("Sales by Video Game Age Rating") + 
  theme_classic() # Notice we Have Three Ratings with No Data (We'll Remove Below)

### Relationship Between Critic Score/Critic Count and User Score/User Count

plot3 <- vgs2 %>% ggplot(aes(Critic_Count, Critic_Score)) + stat_binhex() + 
  scale_fill_gradientn(colors = c("black", "orange"), name = "Count") + 
  labs(x = "Critic Count", y = "Critic Score") + ggtitle("Critic Score per Critic Count") + 
  theme_classic()

plot4 <- vgs2 %>% ggplot(aes(User_Count, User_Score)) + stat_binhex() + 
  scale_fill_gradientn(colors = c("black", "orange"), name = "Count") + 
  labs(x = "User Count", y = "User Score") + ggtitle("User Score per User Count") + 
  theme_classic()

# For Easy Visual Comparison 

grid.arrange(plot3, plot4, nrow=1, ncol=2)

### Correlation Between Critic and User Scores

vgs2 %>% ggplot(aes(Critic_Score, User_Score)) + stat_binhex() + 
  scale_fill_gradientn(colors = c("black", "orange"), name = "Count") + 
  ggtitle("Correlation Between Critic and User Scores") + theme_classic()

cor(vgs2$User_Score, vgs2$Critic_Score, use = "complete.obs")

## Scores Per Genre Comparison

plot9 <- vgs2 %>% select(Genre, User_Score) %>% ggplot(aes(Genre, User_Score, color = Genre)) + 
  geom_jitter(alpha = .3) + labs("Video Game Genre", "User Score") + ggtitle("User Score Per Genre") + 
  theme_classic() + theme(axis.text.x = element_text(angle = 90))

plot10 <- vgs2 %>% select(Genre, Critic_Score) %>% ggplot(aes(Genre, Critic_Score, color = Genre)) +  
  geom_jitter(alpha = .3) + labs("Video Game Genre", "Critic Score") + 
  ggtitle("Critic Score Per Genre") + theme_classic() + theme(axis.text.x = element_text(angle = 90))

# For Easy Visual Comparison 

grid.arrange(plot9, plot10, nrow = 1, ncol = 2)

## Critic vs. User Scores Scaled

# Let's Save a New Variable to Protect Against Corrupting the Original

vgs3 <- vgs2

# Let's Do Some More Comparison Between Critic and User Scores
# We Should Scale the User Score to Make it Easier to Compare to Critics (NOTE: Critics had a 0:100 scale; Users used a 0:10 scale)

vgs3$User_Score_Scaled <- as.numeric(as.character(vgs3$User_Score)) * 10

# Now, Let's Compare the Critic and User Scores Once Again

vgs3 %>% ggplot(aes(Critic_Score, User_Score_Scaled)) + geom_point(aes(color = Platform)) + 
  geom_smooth(method = "lm", size = 1) + labs(x = "Critic Score", y = "User Score") + 
  ggtitle("Comparison of Critic and User Scores on Equal Scales") + theme_classic()

## Let's Compare Critic and User Scores Through Ratings and Systems

vgs3 %>% filter(Rating %in% c("E", "E10+", "M", "T")) %>% 
  ggplot(aes(Critic_Score, User_Score_Scaled)) + geom_point(aes(color = Platform)) + 
  geom_smooth(method = "lm", size = 1) + facet_wrap(~Rating) + 
  labs(x = "Critic Score", y = "User Score") + 
  ggtitle("Comparison of Critic and User Scores on Equal Scales") + theme_classic()


### Global Sales Comparison with Regional Sales

plot6 <- vgs2 %>% select(Global_Sales, NA_Sales, Genre) %>% 
  ggplot(aes(Global_Sales, NA_Sales, color = Genre)) + stat_smooth(method = "lm") + 
  labs(x = "Global Sales", y = "NA Sales") + ggtitle("GS vs. NA Sales across Genres") + 
  theme_classic()

plot7 <- vgs2 %>% select(Global_Sales, JP_Sales, Genre) %>% 
  ggplot(aes(Global_Sales, JP_Sales, color = Genre)) + stat_smooth(method = "lm") + 
  labs(x = "Global Sales", y = "JP Sales") + ggtitle("GS vs. JP Sales across Genres") + 
  theme_classic()

# For Easy Visual Comparison 

grid.arrange(plot6, plot7, nrow=1, ncol=2)

# Data Analysis #

### Now to Figure Out How Many Video Games Are Released Each Year

vgs_per_year <- vgs2 %>% select(Name, Genre, Year_of_Release, Rating) %>% 
  group_by(Year_of_Release, Genre) %>% summarize(gamescount = n())

vgs_per_year %>% ggplot(aes(x = Year_of_Release, y = gamescount, group = Genre, color = Genre)) + 
  stat_sum(size = 1) + geom_line() + labs(x = "Year Video Game Released", y = "Number of Games Released") + 
  ggtitle("Number of Games Released by Year") + theme_classic() + 
  theme(axis.text.x = element_text(angle = 90))

## What are the Annual Video Games Sales?

vgs2 %>% select(Year_of_Release, Global_Sales) %>% group_by(Year_of_Release) %>% 
  summarize(GS = sum(Global_Sales)) %>% 
  ggplot(aes(Year_of_Release, GS, fill = Year_of_Release, color = Year_of_Release)) + 
  geom_area() + geom_point() + labs(x = "Year of Release", y = "Total Sales") + ggtitle("Total Sales by Year") + 
  theme_classic() + theme(axis.text.x = element_text(angle = 90))

## Top Video Game Companies

vgs2 %>% select(Publisher, Global_Sales) %>% group_by(Publisher) %>% summarize(GS = sum(Global_Sales)) %>% 
  arrange(desc(GS)) %>% head(10) %>% ggplot(aes(Publisher, GS, fill = Publisher)) + geom_bar(stat="identity") + 
  labs(x = "Year of Release", y = "Total Sales") + ggtitle("Global Sales by Company") + theme_classic() + 
  theme(axis.text.x = element_text(angle = 90))

## Top Video Game Systems

systems <- vgs2 %>% select(Platform, Global_Sales) %>% group_by(Platform) %>% 
  summarize(GS = sum(Global_Sales))
systems$market_share <- round(systems$GS / sum(systems$GS) * 100) # Calculating Market Share of Each System
systems$market_share <- sort(systems$market_share) # Simply Sorting it for Ease and Clarity

systems %>% ggplot(aes(Platform, market_share, fill = Platform)) + geom_bar(stat = "identity") + 
  labs(x = "Video Game System", y = "Total Sales") + ggtitle("Total Sales by Video Game System") + 
  theme_classic() + theme(axis.text.x = element_text(angle = 90))

## Best-Selling Video Games 

vgs2 %>% select(Name, Global_Sales) %>% group_by(Name) %>% 
  summarize(GS = sum(Global_Sales)) %>% arrange(desc(GS)) %>% 
  head(10) %>% kable(col.names = c("Video Game Name", "Global Sales"))

## Top 10 Best-Rated Games by Users

vgs2 %>% select(Name, User_Score) %>% group_by(Name) %>% summarize(meanus = mean(User_Score)) %>% 
  arrange(desc(meanus)) %>% head(10) %>% kable(caption = "Top 10 Rated Games by Users", col.names = c("Video Game Name", "Avg User Score"))

## Top 10 Best-Rated Games by Critics

vgs2 %>% select(Name, Critic_Score) %>% group_by(Name) %>% summarize(meancs = mean(Critic_Score)) %>% 
  arrange(desc(meancs)) %>% head(10) %>% kable(caption = "Top 10 Rated Games by Critics", col.names = c("Video Game Name", "Avg Critic Score"))

## Compounded Effect: Genre and System?

vgs2 %>% select(Platform, Genre, Global_Sales) %>% group_by(Platform, Genre) %>% 
  summarize(GS = sum(Global_Sales)) %>% ggplot(aes(Genre, Platform, fill = GS)) + 
  geom_tile() + scale_fill_gradientn(colors = c("black", "orange")) + 
  labs(x = "Genre", y = "Video Game System") + ggtitle("Genre + System Compounded Effect?") +
  theme_classic() + theme(legend.position = "none", axis.text.x = element_text(angle = 90))

## Popularity Due to Rating?

vgs2 %>% select(Year_of_Release, Global_Sales, Rating) %>% 
  filter(Rating %in% c("E", "E10+", "M", "T")) %>% group_by(Year_of_Release, Rating) %>% 
  summarize(GS = sum(Global_Sales)) %>% arrange(desc(GS)) %>%
  ggplot(aes(Year_of_Release, GS, group = Rating, fill = Rating)) + 
  geom_bar(stat = "identity", position = "fill") + scale_y_continuous(labels = percent) +
  labs(x = "Years of Release", y = "Percentage of Games Sold by Rating") + 
  ggtitle("Video Games Sold by Maturity Ratings for Certain Years") + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90))

## Overall Sales by Regions

vgs2 %>% select(Year_of_Release, NA_Sales, JP_Sales, EU_Sales, Other_Sales) %>% 
  group_by(Year_of_Release) %>% 
  summarize(NAS = sum(NA_Sales), JP = sum(JP_Sales), EU = sum(EU_Sales), Other = sum(Other_Sales)) %>% 
  ggplot(aes(Year_of_Release, NAS, group = 1, color = "NAS")) + 
  geom_line() + geom_line(aes(y = JP, color = "JP")) + geom_line(aes(y = EU, color = "EU")) + 
  geom_line(aes(y = Other, color = "Other")) + labs(x = "Year of Release", y = "Total Sales") + 
  ggtitle("Sales of Video Games by Region") + theme_classic() + theme(axis.text.x = element_text(angle = 90))

## Predicting Sales as a Function of Critic Scores

# Let's Look at the Three Major Video Game Companies - Nintendo, Microsoft, and Sony

# Microsoft

microsoft_plot <- vgs3 %>% filter(vgs3$companies == 'Microsoft') %>% 
  ggplot(aes(Critic_Score, NA_Sales)) + geom_point(aes(color = Genre)) + 
  ylim(0, 5) + geom_smooth() + labs(x = "Critic Score", y = "Total Sales") + 
  ggtitle("Microsoft") + theme_classic() + theme(axis.text.x = element_blank())

# Nintendo

nintendo_plot <- vgs3 %>% filter(vgs3$companies == 'Nintendo') %>% 
  ggplot(aes(Critic_Score, Global_Sales)) + geom_point(aes(color = Genre)) + 
  ylim(0, 5) + geom_smooth() + labs(x = "Critic Score", y = "Total Sales") + 
  ggtitle("Nintendo") + theme_classic() + theme(axis.text.x = element_blank())

# Sony

sony_plot <- vgs3 %>% filter(vgs3$companies == 'Sony') %>% ggplot(aes(Critic_Score, NA_Sales)) + 
  geom_point(aes(color = Genre)) + ylim(0, 5) + geom_smooth() + 
  labs(x = "Critic Score", y = "Total Sales") + ggtitle("Sony") + theme_classic() + 
  theme(axis.text.x = element_blank())

# Let's Compare those 3 Visually Together

ggarrange(nintendo_plot, microsoft_plot, sony_plot, ncol = 3)

#### Modeling ####

## Model 1: Simple Prediction

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings) ^ 2, na.rm = T))
}

val_ind <- createDataPartition(vgs3$Global_Sales, times = 1, p = 0.3, list = FALSE)
train_set <- vgs3[-val_ind,]
val_set <- vgs3[val_ind,]

# Simple Prediction Mean

mu <- mean(train_set$Global_Sales)

# What was the Mu?

mu

# Simple Prediction

rmse_model1 <- RMSE(train_set$Global_Sales, mu)
rmse_preds <- data.frame(Method = "Model 1: Simple Prediction Model", RMSE = rmse_model1)

# Check Results

rmse_preds

# Model 2: Generalized Linear Model

train_glm <- train(Global_Sales ~ Platform + User_Score + Critic_Score + Critic_Count + Genre + Year_of_Release + Rating, method = "glm", data = train_set, na.action = na.exclude)

rmse_model2 <- getTrainPerf(train_glm)$TrainRMSE
rmse_preds <- add_row(rmse_preds, Method = "Model 2: Generalized Linear Model", RMSE = rmse_model2)

# Check Results

rmse_preds

## Model 3: Knn 

train_knn <- train(Global_Sales ~ Platform + User_Score + Critic_Score + Critic_Count + Genre + Year_of_Release + Rating, method = "knn", data = train_set, na.action = na.exclude)

rmse_model3 <- getTrainPerf(train_knn)$TrainRMSE
rmse_preds <- add_row(rmse_preds, Method = "Model 3: K-Nearest Neighbors Model", RMSE = rmse_model3)

# Check Results

rmse_preds

## Model 4: Support Vector Machines 

train_svm <- train(Global_Sales ~ Platform + User_Score + Critic_Score + Critic_Count + Genre + Year_of_Release + Rating, method = "svmLinear", data = train_set, na.action = na.exclude, scale = FALSE)

rmse_model4 <- getTrainPerf(train_svm)$TrainRMSE
rmse_preds <- add_row(rmse_preds, Method = "Model 4: Support Vector Machines Model", RMSE = rmse_model4)

# Check Results

rmse_preds

## Model 5: Random Forest

train_rf <- train(Global_Sales ~ Platform + User_Score + Critic_Score + Critic_Count + Genre + Year_of_Release + Rating, method = "rf", data = train_set, na.action = na.exclude, ntree = 40, metric="RMSE", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3), importance = T)

rmse_model5 <- getTrainPerf(train_rf)$TrainRMSE
rmse_preds <- add_row(rmse_preds, Method = "Model 5: Random Forest", RMSE = rmse_model5)

# Check Results

rmse_preds


## Variable Importance Per Model

varImp(train_glm)
varImp(train_knn)
varImp(train_svm)
varImp(train_rf)

#### Results ####

## Final Model on Validation Set

train_final <- train(Global_Sales ~ Platform + User_Score + Critic_Score + Critic_Count + Genre + Year_of_Release + Rating, 
    method = "rf", data = train_set, na.action = na.exclude, ntree = 40, metric="RMSE", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3), importance = T)

predicted <- predict(train_final, newdata = val_set)
rmse_model6 <- RMSE(val_set$Global_Sales, predicted)
rmse_preds <- add_row(rmse_preds, Method = "Model 6: Final on Validation", RMSE = rmse_model6)

## Variable Importance Final Model

varImp(train_final)

# Check Final Results

rmse_preds

#### Conclusion ####

print("We were able to get the RMSE down! This shows that the model works pretty well as a machine learning algorithm to predict Global Sales based on VGChartz' Video Games Sales Dataset. The optimal model thus far is based on the Random Forest Model. However, I had to remove two variables in the final RF model due to returned errors - I could not debug in time but figured out that removing the variables did provide at least a good model. Another limitation was hardware - this was difficult to manage even with the parallel processing. My final report took roughly 25 mins to produce with parallel processing, but was successful nonetheless. Future work would be to further tune the RF model and spend more time finding optimal numbers for ntree, mtry, and variables to use. This RMSE is still much larger than desired but was successful in improving our algorithm by 23%. For reference, our worst model - the Simple Prediction Model - received an RMSE of 2.026681.  It would be beneficial if we could garner more complete data as well through web scraping. One model I wished I had the time to adopt at the end was Matrix Factorization - I think this would be beneficial for future research.")

## Admin Note

stopCluster(cl) # Stops Parallel Processing

