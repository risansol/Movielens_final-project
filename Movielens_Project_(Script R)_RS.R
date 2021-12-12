##===
# Downloading and preparing the data set, as presented in section "Movielens Overview"
##===


if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]   #train set
temp <- movielens[test_index,]   

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")  #test set

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##====
# Now, we are ready to elaborate and evaluate the different regression models
##====

## remember!!===>    edx <-train set  & validation <- test set   ###

##Movielens_data exploration

movielens %>% as_tibble()

movielens %>% 
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

# rating per movie

library(gridExtra)
movie_plot <- movielens %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "grey") + 
  scale_x_log10() + 
  ggtitle("Movies")


# rating per user
user_plot <- movielens %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "grey") + 
  scale_x_log10() + 
  ggtitle("Users") 

grid.arrange(movie_plot, user_plot, ncol=2)
##--function that computes the RMSE for vector of ratings and their corresponding predictors

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##-- 1.-first model ---> just the estimating rating average (p641)

mu <- mean(edx$rating)
mu

##-- 2.- movie effects

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))
# predicted ratings
predicted_rating_b_i <- mu + validation %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  pull(b_i)

##-- 3.- movie+user effects

user_avgs <- edx %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating - mu - b_i))
# predicted ratings
predicted_rating_b_u <- validation %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)



#####calculate RMSE for average rating, movie, user and time effect models

rmse_model_mu <- RMSE(validation$rating, mu)
rmse_model_movie <- RMSE(validation$rating, predicted_rating_b_i)
rmse_model_user <- RMSE(validation$rating,predicted_rating_b_u)
rmse_model_time <- RMSE(validation$rating,predicted_rating_b_u_t)


##--4.- Regularization 

## top 10 worst and 10 best movies based on b_i, and how often they are rated
movie_titles <- movielens %>% 
  select(movieId, title) %>% 
  distinct()

### top 10 best
movie_avgs %>% 
  left_join(movie_titles, by='movieId') %>% 
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(title)

###-- how often they are rated

edx %>% count(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(movie_titles, by='movieId') %>% 
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

### top 10 worst
movie_avgs %>% 
  left_join(movie_titles, by='movieId') %>% 
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(title)

###-- how often they are rated
edx %>% count(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(movie_titles, by='movieId') %>% 
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)

#using cross validation to pick up lambda

lambda <- seq(0, 10, 0.25)

##-- 4.1.- Regularized movie effect model (p. 650)

mu <- mean(edx$rating)

summation_term <- edx %>% 
  group_by(movieId) %>% 
  summarise(s = sum(rating - mu), n_i = n())

rmse_reg_b_i <-  sapply(lambda, function(l){
  predicted_rating_reg_b_i  <- validation %>% 
    left_join(summation_term, by = 'movieId') %>% 
    mutate(b_i_lambda = s/(n_i + l)) %>% 
    mutate(pred = mu + b_i_lambda) %>% 
    pull(pred)
  return(RMSE(predicted_rating_reg_b_i, validation$rating))
})

qplot(lambda, rmse_reg_b_i)

#for the full model, the optimal lambda is
lambda[which.min(rmse_reg_b_i)]

#the minimun error is

rmse_model_reg_b_i <- min(rmse_reg_b_i)

##-- 4.2.- Regularized movie + user effect model

rmse_reg_b_i_b_u <- sapply(lambda, function(l){
  b_i_lambda <- edx %>% 
    group_by(movieId) %>% 
    summarise(b_i_lambda = sum(rating - mu)/(n()+l))
  
  
  b_u_lambda <- edx %>% 
    left_join(b_i_lambda, by='movieId') %>%
    group_by(userId) %>% 
    summarise(b_u_lambda = sum(rating - mu - b_i_lambda)/(n()+l))
  
  predicted_rating_reg_b_i_b_u <- validation %>% 
    left_join(b_i_lambda, by='movieId') %>% 
    left_join(b_u_lambda, by='userId') %>% 
    mutate(pred = mu + b_i_lambda + b_u_lambda) %>% 
    pull(pred)
  
  return(RMSE(predicted_rating_reg_b_i_b_u, validation$rating))
})


qplot(lambda, rmse_reg_b_i_b_u)

#for the full model, the optimal lambda is

lambda[which.min(rmse_reg_b_i_b_u)]

#the minimun error is
rmse_model_reg_b_i_b_u <- min(rmse_reg_b_i_b_u)

###table with the summary of all rmse results

rmse_result <- data.frame(model  = c("Average", "Movie Effect", "Movie + User Effect",
                                     "Movie+ User + Time effect", "Regularized Movie Effect",
                                     "Regularized Movie + User Effect"), 
                          RMSE = round (c(rmse_model_mu,rmse_model_movie,rmse_model_user,
                                          rmse_model_time,rmse_model_reg_b_i,
                                          rmse_model_reg_b_i_b_u),4))

rmse_result




