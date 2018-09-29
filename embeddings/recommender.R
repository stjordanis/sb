#reticulate::use_condaenv("tf-12-gpu-0924")
reticulate::use_condaenv("tf-11-gpu")
#reticulate::use_condaenv("tf-10-gpu")

library(keras)
library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(stringr)
library(Rtsne)
library(lattice)

data_dir <- "ml-latest-small"

movies <- read_csv(file.path(data_dir, "movies.csv"))
movies %>% group_by(genres) %>% summarise(count = n()) %>% arrange(desc(count)) %>% print(n = 100)
# 1 Drama                                1170
# 2 Comedy                                809
# 4 Documentary                           365
# 8 Horror                                183
# 15 Thriller                               74
# 33 Action                                 39
# 46 Western                                31
# 83 Musical                                18
# 85 Sci-Fi                                 17
# 95 Romance                                14

movies <- movies %>% mutate(main_genre =
                              if_else(
                                str_detect(genres, "Drama"),
                                "Drama",
                                if_else(
                                  str_detect(genres, "Comedy"),
                                  "Comedy",
                                  if_else(
                                    str_detect(genres, "Documentary"),
                                    "Documentary",
                                    if_else(
                                      str_detect(genres, "Horror"),
                                      "Horror",
                                      if_else(
                                        str_detect(genres, "Thriller"),
                                        "Thriller",
                                        "other")
                                              )
                                            )
                                          )
                                        )
                                      )

ratings <- read_csv(file.path(data_dir, "ratings.csv"))
dense_movies <- ratings %>% select(movieId) %>% distinct() %>% rowid_to_column()
ratings <- ratings %>% inner_join(dense_movies) %>% rename(movieIdDense = rowid)
ratings <- ratings %>% inner_join(movies) %>% select(userId, movieIdDense, rating, title, main_genre, genres)

n_movies <-
  ratings %>% select(movieIdDense) %>% distinct() %>% nrow()
n_users <- ratings %>% select(userId) %>% distinct() %>% nrow()

train_indices <- sample(1:nrow(ratings), 0.8 * nrow(ratings))
train_ratings <- ratings[train_indices,]
valid_ratings <- ratings[-train_indices,]


train_ratings %>% select(movieIdDense) %>% distinct() %>% nrow()
train_ratings %>% select(userId) %>% distinct() %>% nrow()
valid_ratings %>% select(movieIdDense) %>% distinct() %>% nrow()
valid_ratings %>% select(userId) %>% distinct() %>% nrow()

x_train <-
  train_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_train <- train_ratings %>% select(rating) %>% as.matrix()
x_valid <-
  valid_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_valid <- valid_ratings %>% select(rating) %>% as.matrix()

embedding_dim <- 64


# Simple dot product model ------------------------------------------------

simple_dot <- function(embedding_dim,
                       n_users,
                       n_movies,
                       name = "simple_dot") {
  keras_model_custom(name = name, function(self) {
    self$user_embedding <-
      layer_embedding(
        input_dim = n_users + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = "user_embedding"
      )
    self$movie_embedding <-
      layer_embedding(
        input_dim = n_movies + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = "movie_embedding"
      )
    
    # https://github.com/keras-team/keras/issues/6151
    self$dot <-
      layer_lambda(
        f = function(x) {
          k_batch_dot(x[[1]], x[[2]], axes = 2)
        }
      )
    
    function(x, mask = NULL) {
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <- self$user_embedding(users)
      movie_embedding <- self$movie_embedding(movies)
      self$dot(list(user_embedding, movie_embedding))
    }
    
  })
}

model <- simple_dot(embedding_dim, n_users, n_movies)
model %>% compile(loss = "mse",
                  optimizer = "adam")

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)
plot(history)


# Dot product with biases ----------------------------------------------------------------

ratings %>% summarise(max_rating = max(rating),
                      min_rating = min(rating))
max_rating <- 5
min_rating <- 0.5

dot_with_bias <- function(embedding_dim,
                          n_users,
                          n_movies,
                          max_rating,
                          min_rating,
                          name = "dot_with_bias") {
  keras_model_custom(name = name, function(self) {
    self$user_embedding <-
      layer_embedding(input_dim = n_users + 1,
                      output_dim = embedding_dim,
                      name = "user_embedding")
    self$movie_embedding <-
      layer_embedding(input_dim = n_movies + 1,
                      output_dim = embedding_dim,
                      name = "movie_embedding")
    self$user_bias <-
      layer_embedding(input_dim = n_users + 1,
                      output_dim = 1,
                      name = "user_bias")
    self$movie_bias <-
      layer_embedding(input_dim = n_movies + 1,
                      output_dim = 1,
                      name = "movie_bias")
    self$user_dropout <- layer_dropout(rate = 0.3)
    self$movie_dropout <- layer_dropout(rate = 0.6)
    self$dot <-
      layer_lambda(
        f = function(x)
          k_batch_dot(x[[1]], x[[2]], axes = 2),
        name = "dot"
      )
    self$dot_bias <-
      layer_lambda(
        f = function(x)
          k_sigmoid(x[[1]] + x[[2]] + x[[3]]),
        name = "dot_bias"
      )
    self$pred <- layer_lambda(
      f = function(x)
        x * (self$max_rating - self$min_rating) + self$min_rating,
      name = "pred"
    )
    self$max_rating <- max_rating
    self$min_rating <- min_rating
    
    function(x, mask = NULL) {
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <-
        self$user_embedding(users) %>% self$user_dropout()
      movie_embedding <-
        self$movie_embedding(movies) %>% self$movie_dropout()
      dot <- self$dot(list(user_embedding, movie_embedding))
      dot_bias <-
        self$dot_bias(list(dot, self$user_bias(users), self$movie_bias(movies)))
      pred <- self$pred(dot_bias)
      pred
    }
    
  })
}

model <- dot_with_bias(embedding_dim, n_users,
                       n_movies,max_rating, min_rating)
model %>% compile(loss = "mse",
                  optimizer = "adam")

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)

plot(history)

user_embeddings <-
  (model %>% get_layer("user_embedding") %>% get_weights())[[1]]
dim(user_embeddings)
user_embeddings

movie_embeddings <-
  (model %>% get_layer("movie_embedding") %>% get_weights())[[1]]
dim(movie_embeddings)
movie_embeddings

user_bias <-
  (model %>% get_layer("user_bias") %>% get_weights())[[1]]
dim(user_bias)
user_bias

movie_bias <-
  (model %>% get_layer("movie_bias") %>% get_weights())[[1]]
dim(movie_bias)
movie_bias


# Embedding model ----------------------------------------------------------------

embedding_model <- function(embedding_dim,
                            n_users,
                            n_movies,
                            max_rating,
                            min_rating,
                            name = "simple_dot") {
  keras_model_custom(name = name, function(self) {
    self$user_embedding <-
      layer_embedding(input_dim = n_users + 1,
                      output_dim = embedding_dim,
                      #     embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
                      name = "user_embedding")
    self$movie_embedding <-
      layer_embedding(input_dim = n_movies + 1,
                      output_dim = embedding_dim,
                      #     embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
                      name = "movie_embedding")
    self$concat <-
      layer_lambda(
        f = function(x)
          k_concatenate(x),
        name = "dot"
      )
    self$dense1 <- layer_dense(units = 10, activation = "relu")
    self$dense2 <- layer_dense(units = 10, activation = "sigmoid")
    self$dropout1 <- layer_dropout(rate = 0.5)
    self$dropout2 <- layer_dropout(rate = 0.5)
    self$pred <- layer_lambda(
      f = function(x)
        x * (self$max_rating - self$min_rating + 1) + self$min_rating - 0.5,
      name = "pred"
    )
    self$max_rating <- max_rating
    self$min_rating <- min_rating
    
    function(x, mask = NULL) {
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <- self$user_embedding(users)
      movie_embedding <- self$movie_embedding(movies)
      concat <- self$concat(list(user_embedding, movie_embedding))
      x <- self$dropout1(concat) %>% self$dense1() %>%
        self$dropout2() %>% self$dense2() %>% self$pred()
      x
    }
    
  })
}

model <- embedding_model(embedding_dim, n_users,
                         n_movies, max_rating, min_rating)
model %>% compile(loss = "mse",
                  optimizer = "adam",
                  metrics = "mse")

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)
# 0.7859
plot(history)


# Embeddings PCA -----------------------------------------------------------


user_bias[, 1] %>% data_frame(bias = .) %>% ggplot(aes(x = bias)) + geom_density()
movie_bias[, 1] %>% data_frame(bias = .) %>% ggplot(aes(x = bias)) + geom_density()

# > movie_embeddings[1:5,1:5] * 100
# [,1]      [,2]      [,3]       [,4]        [,5]
# [1,]  -1.625595  3.792311 -2.388493   3.414843  -4.5218755
# [2,] -27.741063 -7.445961 13.293166 -18.429738 -16.2929922
# [3,] -14.812906  3.768364 -6.703106  15.442163 -11.4187367
# [4,]  -8.068993 17.547150 24.357036  15.565166   0.2887363
# [5,]   4.365638  5.871716  4.775926   2.753527  21.6409311

image(
  1:64,
  1:20,
  t(movie_embeddings[1:20, 1:64]) * 127.5 + 127.5,
  col = viridis(10),
  #col = gray((0:255) / 255),
  xaxt = 'n',
  yaxt = 'n',
  xlab = "movies",
  ylab = "components",
  main = "movie embeddings"
)

levelplot(t(movie_embeddings[1:20, 1:64]),
          xlab = "",
          ylab = "",
          scale = (list(draw = FALSE)))

movie_pca <- movie_embeddings %>% prcomp(center = FALSE)
plot(movie_pca)
components <- movie_pca$x %>% as.data.frame() %>% rowid_to_column()
components
sdev <- movie_pca$sdev

user_pca <- user_embeddings %>% prcomp(center = FALSE)
plot(user_pca)

ratings_with_pc12 <-
  ratings %>% inner_join(components %>% select(rowid, PC1, PC2), by = c("movieIdDense" = "rowid"))
ratings_grouped <-
  ratings_with_pc12 %>% group_by(title) %>% summarize(
    PC1 = max(PC1),
    PC2 = max(PC2),
    rating = mean(rating),
    genres = max(genres),
    num_ratings = n()
  )
ratings_grouped %>% filter(num_ratings > 10) %>% arrange(PC1) %>% print(n = 20)
ratings_grouped %>% filter(num_ratings > 10) %>% arrange(desc(PC1)) %>% print(n = 20)

ratings_with_emb <- ratings %>% inner_join(movie_embeddings %>% as.data.frame() %>% rowid_to_column(), by = c("movieIdDense" = "rowid")) 

tsne_input <- ratings_with_emb %>% select(starts_with("V")) %>% as.matrix()

for (perplexity in c(5, 10, 20, 30, 50)) {
  tsne <-
    Rtsne(
      tsne_input,
      check_duplicates = FALSE,
      perplexity = perplexity,
      verbose = TRUE,
      pca = FALSE,
      max_iter = 1000
    )
  png(paste0("perp_", perplexity, ".png"))
  plot(
    tsne$Y,
    col = ratings_with_all_pcs$main_genre %>% as.factor() %>% as.numeric(),
    xlab = "",
    ylab = "",
    xaxt = 'n',
    yaxt = 'n',
    main = paste0("perplexity: ", perplexity)
  )
  dev.off()
}

# Do it yourself -----------------------------------------------------------

SimpleEmbedding <- R6::R6Class(
  "SimpleEmbedding",
  
  inherit = KerasLayer,
  
  public = list(
    output_dim = NULL,
    emb_input_dim = NULL,
    embeddings = NULL,
    
    initialize = function(emb_input_dim, output_dim) {
      self$emb_input_dim <- emb_input_dim
      self$output_dim <- output_dim
    },
    
    # input_shape is NULL
    build = function(input_shape) {
      self$embeddings <- self$add_weight(
        name = 'embeddings',
        shape = list(self$emb_input_dim, self$output_dim),
        initializer = initializer_random_uniform(),
        trainable = TRUE
      )
    },
    
    call = function(x, mask = NULL) {
      x <- k_cast(x, "int32") 
      x <- k_gather(self$embeddings, x)
      x
    },
    
    compute_output_shape = function(input_shape) {
      list(self$output_dim)
    }
  )
)

layer_simple_embedding <-
  function(object,
           emb_input_dim,
           output_dim,
           name = NULL,
           trainable = TRUE) {
    create_layer(
      SimpleEmbedding,
      object,
      list(
        emb_input_dim = as.integer(emb_input_dim),
        output_dim = as.integer(output_dim),
        name = name,
        trainable = trainable
      )
    )
  }

# simple_embedding <- function(emb_input_dim,
#                              output_dim,
#                         name = "simple_dot") {
#   keras_model_custom(name = name, function(self) {
#     
#     # with both implementations
#     # AttributeError: 'RModel' object has no attribute '_trainable_weights'
#     # self$embeddings <- self$add_weight(
#     #   name = 'embeddings',
#     #   shape = list(emb_input_dim, output_dim),
#     #   initializer = initializer_random_uniform())
#     
#     #   NotImplementedError: `add_variable` is not supported on Networks.
#     # self$embeddings <- self$add_variable(
#     #   name = 'embeddings', 
#     #   shape = list(emb_input_dim, output_dim),
#     #   initializer = initializer_random_uniform())
#     
#     # only gets optimized when using implementation = tf
#     self$embeddings <- k_variable(
#       k_random_uniform(shape = shape(emb_input_dim, output_dim)),
#       name = "embeddings"
#     )
#     
#     function(x, mask = NULL) {
#       x <- k_cast(x, "int32")
#       x <- k_gather(self$embeddings, x)
#       x
#     }
#     
#   })
# }

simple_dot2 <- function(embedding_dim,
                        n_users,
                        n_movies,
                        name = "simple_dot2") {
  
  keras_model_custom(name = name, function(self) {
    
    self$embedding_dim <- embedding_dim
    
     self$user_embedding <-
       layer_simple_embedding(
         emb_input_dim = list(n_users + 1),
         output_dim = embedding_dim,
         name = "user_embedding"
       )
     self$movie_embedding <-
       layer_simple_embedding(
         emb_input_dim = list(n_movies + 1),
         output_dim = embedding_dim,
         name = "movie_embedding"
       )
    
    #self$user_embedding <- simple_embedding(as.integer(n_users), as.integer(embedding_dim))
    #self$movie_embedding <- simple_embedding(as.integer(n_movies), as.integer(embedding_dim))
    self$dot <-
      layer_lambda(output_shape = self$embedding_dim,
        f = function(x) {
          k_batch_dot(x[[1]], x[[2]], axes = 2)
        }
      )
    
    function(x, mask = NULL) {
     
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <- self$user_embedding(users)
      movie_embedding <- self$movie_embedding(movies)
      self$dot(list(user_embedding, movie_embedding))
    }
    
  })
}

model <- simple_dot2(embedding_dim, n_users, n_movies)

model %>% compile(loss = "mse",
                  optimizer = "adam"
                  )


history <- model %>% fit(
  #x_train,
  #y_train,
  x_train[1:2, ],
  y_train[1:2],
  epochs = 20,
  batch_size = 2#,
  #validation_data = list(x_valid, y_valid),
  #callbacks = list(callback_early_stopping(patience = 2))
)
# val_loss: 1.2688
plot(history)

# Testing stuff -----------------------------------------------------------


sess <- k_get_session()
test1 <- k_constant(c(1,2,3,4), shape = c(2,2))
test2 <- k_constant(c(10,20,30,40), shape = c(2,2))
#
test1 <- k_constant(c(1,2,3,4,5,6), shape = c(3,1,2))
test2 <- k_constant(c(10,20,30,40,50,60), shape = c(3,1,2))
#
# sess$run(list(test1, test2, k_batch_dot(test1, test2, axes = 3)))
#
test3 <- k_constant(c(10,20,30,40,50,60,70,80,90), shape = c(3,3))
test3 %>% k_eval()
k_gather(test3, list(0L, 1L)) %>% k_eval()
k_gather(test3, list(1L, 2L)) %>% k_eval()
# 
# test_simple_embedding <- simple_embedding(as.integer(n_users), as.integer(embedding_dim))
# test_simple_embedding(x_train[1:10, 1]) %>% k_eval()
