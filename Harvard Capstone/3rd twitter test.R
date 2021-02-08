
install.packages("tm")
install.packages("gmodels")
install.packages("ROCR")
install.packages("randomForest")
install.packages("wordcloud")
install.packages("syuzhet")
install.packages("tidytext")
install.packages("doParallel")

library(tidytext)
library(tidyverse) # metapackage with lots of helpful functions
library(tm)
library(SnowballC)
library(caTools)
#install.packages("e1071")
library(e1071)
library(caret)
library(gmodels)
library(pROC)
library(ROCR)
library(randomForest)
library(xlsx)
library(wordcloud)
library(syuzhet)
library(tidyr)
library(ggplot2)
library(tidyverse)
library(doParallel)

split <- detectCores()
split # To Make Sure it Worked
cl <- makePSOCKcluster(split)
registerDoParallel(cl)

#files <- list.files("C:/My Desktop/OHR/Projects/Sentiment Analysis White Paper/Twitter")
files <- list.files("C:/My Desktop/OHR/Projects/Sentiment Analysis White Paper/Twitter")
#data <- read.xlsx("C:/My Desktop/OHR/Projects/Sentiment Analysis White Paper/Twitter/twitter tweets small.xlsx", sheetName = "tweets", header = FALSE, stringsAsFactors = FALSE)
data <- read.xlsx("C:/My Desktop/OHR/Projects/Sentiment Analysis White Paper/Twitter/Sentiment.xlsx", sheetName = "Sentiment", stringsAsFactors = FALSE)

head(data)

## Removing Unused Columns From Data Frame

#data <- data[,-c(2,3,4,5)]
str(data)
head(data)

# Shuffling Rows of Data Drame 

data <- data[sample(nrow(data)),]

# Add Appropriate Column Names to Data Frame

#colnames(data)<-c("label","text")

data_1 <- data %>% 
  select(text, sentiment)
head(data_1)

str(data_1)

round(prop.table(table(data_1$sentiment)),2)

library(tm)
## Loading required package: NLP
library(SnowballC)
corpus = VCorpus(VectorSource(data_1$text))
# a snap shot of the first text stored in the corpus
as.character(corpus[[1]])

corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, letters)
corpus <- tm_map(corpus, content_transformer(tolower))
as.character(corpus[[1]])

dtm = DocumentTermMatrix(corpus)
dtm
dim(dtm)

dtm = removeSparseTerms(dtm, 0.999)
dim(dtm)

#Inspecting the the first 10 tweets and the first 15 words in the dataset
inspect(dtm[0:10, 1:15])

freq<- sort(colSums(as.matrix(dtm)), decreasing=TRUE)

findFreqTerms(dtm, lowfreq=60) #identifying terms that appears more than 60times

library(ggplot2)
wf<- data.frame(word=names(freq), freq=freq)
head(wf)

library("wordcloud")
positive <- subset(data_1,sentiment=="Positive")
head(positive)
wordcloud(positive$text, max.words = 5, scale = c(3,0.5))

negative <- subset(data_1,sentiment=="Negative")
head(negative)
wordcloud(negative$text, max.words = 5, scale = c(3,0.5))

neutral <- subset(data_1,sentiment=="Neutral")
head(neutral)
wordcloud(neutral$text, max.words = 5, scale = c(3,0.5))


## Loading required package: RColorBrewer
library("RColorBrewer")
set.seed(1234)
wordcloud(words = wf$word, freq = wf$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
datasetNB <- apply(dtm, 2, convert_count)

dataset <- as.data.frame(as.matrix(datasetNB))

dataset$Class <- data_1$sentiment
dataset$Class <- as.factor(dataset$Class)
str(dataset$Class)

head(dataset)
dim(dataset)

set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.75,0.25),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$Class))
prop.table(table(test_set$Class))

library(randomForest)
rf_classifier = randomForest(x = train_set[-1210],
                             y = train_set$Class,
                             ntree = 300)

rf_classifier

# Predicting the Test set results
rf_pred = predict(rf_classifier, newdata = test_set[-1210])

# Making the Confusion Matrix
library(caret)

confusionMatrix(table(rf_pred,test_set$Class))

library(e1071)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time( classifier_nb <- naiveBayes(train_set, train_set$Class, laplace = 1,
                                         trControl = control,tuneLength = 7) )

nb_pred = predict(classifier_nb, type = 'class', newdata = test_set)

confusionMatrix(nb_pred,test_set$Class)

svm_classifier <- svm(Class~., data=train_set)
svm_classifier

svm_pred = predict(svm_classifier,test_set)

confusionMatrix(svm_pred,test_set$Class)