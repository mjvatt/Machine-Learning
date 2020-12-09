# 1) K-Means Clustering Image Compression

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
              "GGally", "ggplot2", "ggthemes", "gridExtra", "imager", "jpeg", "kableExtra", "knitr", 
              "ldatuning", "magrittr", "mclust", "NbClust", "petro.One", "plotly", "plotrix", "png", 
              "qdap", "qdapTools", "quanteda", "randomForest", "readxl", "reshape", "RColorBrewer", 
              "rlist", "RWeka", "scales", "SentimentAnalysis", "sentimentr", "SnowballC", "stats", 
              "stm", "stringr", "syuzhet", "tensorflow", "tidyr", "tidytext", "tidyverse", "tm", 
              "topicmodels", "viridisLite", "wordcloud", "xlsx", "zoo")

install(packages)



# Call to load the installed packages also commented out but left in the script in case it was
# needed to perform the script. 



# Call the installed packages

library(dplyr) # dplyr needed for efficient loading of loadApp()

loadApp <- function() {

  my_library <- c("caTools", "car", "caret", "cluster", "Clustering", "corpus", "corrplot", "data.table",
                  "dendextend", "doParallel", "dplyr", "e1071", "factoextra", "FactoMineR", "fpc",
                  "GGally", "ggplot2", "ggthemes", "gridExtra", "imager", "jpeg", "kableExtra", "knitr", 
                  "ldatuning", "magrittr", "mclust", "NbClust", "petro.One", "plotly", "plotrix", "png", 
                  "qdap", "qdapTools", "quanteda", "randomForest", "readxl", "reshape", "RColorBrewer", 
                  "rlist", "RWeka", "scales", "SentimentAnalysis", "sentimentr", "SnowballC", "stats", 
                  "stm", "stringr", "syuzhet", "tensorflow", "tidyr", "tidytext", "tidyverse", "tm", 
                  "topicmodels", "viridisLite", "wordcloud", "xlsx", "zoo")

  install.lib <- my_library[!my_library %>% installed.packages()]

  for(lib in install.lib) install.packages(lib, dependencies = TRUE)

  sapply(my_library, require, character = TRUE)

}

loadApp()

# Import Image

# Need to modify filepath if attempting to load image from another location
image <- readPNG("C:/My Desktop/k means/Images/Satellite.png") # Read image into R

class(image) # Ensure data was read in
original_dim <- dim(image) # Keep original dimensions
dim(image) <- c(dim(image)[1]*dim(image)[2],3) # Reshape dimensions
dim(image) # Verify it worked
#paste("Before compression: ", dim_before[1], "x", dim_before[2])

# Compress with 2 centers

n <- 2
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-2.png") # Output compressed image

# Compress with 4 centers

n <- 4
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-4.png") # Output compressed image

# Compress with 8 centers

n <- 8
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-8.png") # Output compressed image

# Compress with 16 centers

n <- 16
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-16.png") # Output compressed image

# Compress with 32 centers

n <- 32
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-32.png") # Output compressed image

# Compress with 64 centers

n <- 64
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-64.png") # Output compressed image

# Compress with 128 centers

n <- 128
kmeans_image <- kmeans(image, centers = n, iter.max = 100) # Run kmeans clustering algorithm
str(kmeans_image) # Ensure kmeans() executed properly
img <- kmeans_image$centers[kmeans_image$cluster,] # Retrieve colors
dim(img) <- original_dim # Reshape back to original dimensions
writePNG(img, "C:/My Desktop/k means/Images/Satellite-compressed-128.png") # Output compressed image
 






