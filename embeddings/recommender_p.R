library(keras)
library(readr)
library(dplyr)
library(lattice)
library(tibble)


# Do it yourself embeddings -----------------------------------------------------------

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



# Embeddings for collaborative filtering: Simple dot product --------------

simple_dot <- function(embedding_dim,
                       n_users,
                       n_movies,
                       name = "simple_dot") {
  keras_model_custom(name = name, function(self) {
    self$user_embedding <-
      layer_embedding(
        input_dim = n_users + 1,
        output_dim = embedding_dim,
        name = "user_embedding"
      )
    self$movie_embedding <-
      layer_embedding(
        input_dim = n_movies + 1,
        output_dim = embedding_dim,
        name = "movie_embedding"
      )
    
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


data_dir <- "ml-latest-small"

movies <- read_csv(file.path(data_dir, "movies.csv"))
ratings <- read_csv(file.path(data_dir, "ratings.csv"))
dense_movies <-
  ratings %>% select(movieId) %>% distinct() %>% rowid_to_column()
ratings <-
  ratings %>% inner_join(dense_movies) %>% rename(movieIdDense = rowid)
ratings <-
  ratings %>% inner_join(movies) %>% select(userId, movieIdDense, rating, title, genres)

n_movies <-
  ratings %>% select(movieIdDense) %>% distinct() %>% nrow()
n_users <- ratings %>% select(userId) %>% distinct() %>% nrow()

train_indices <- sample(1:nrow(ratings), 0.8 * nrow(ratings))
train_ratings <- ratings[train_indices, ]
valid_ratings <- ratings[-train_indices, ]

x_train <-
  train_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_train <- train_ratings %>% select(rating) %>% as.matrix()
x_valid <-
  valid_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_valid <- valid_ratings %>% select(rating) %>% as.matrix()

embedding_dim <- 64

model <- simple_dot(embedding_dim, n_users, n_movies)
model %>% compile(loss = "mse",
                  optimizer = "adam")

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)

png("simple_embedding.png", width = 680, height = 400)
plot(history, metrics = "loss")
dev.off()


# Dot product with biases ----------------------------------------------------------------

max_rating <-
  ratings %>% summarise(max_rating = max(rating)) %>% pull()
min_rating <-
  ratings %>% summarise(min_rating = min(rating)) %>% pull()

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
                  optimizer = "adam"
                  )

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)

png("bias_embedding.png", width = 680, height = 400)
plot(history, metrics = "loss")
dev.off()

# A closer look at embeddings --------------

movie_embeddings <-
  (model %>% get_layer("movie_embedding") %>% get_weights())[[1]]
dim(movie_embeddings)

levelplot(t(movie_embeddings[2:21, 1:64]),
          xlab = "",
          ylab = "",
          scale = (list(draw = FALSE)))

movie_pca <- movie_embeddings %>% prcomp(center = FALSE)
plot(movie_pca)
components <- movie_pca$x %>% as.data.frame() %>% rowid_to_column()

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
ratings_grouped %>% filter(num_ratings > 20) %>% arrange(PC1) %>% print(n = 10)
ratings_grouped %>% filter(num_ratings > 20) %>% arrange(desc(PC1)) %>% print(n = 10)


# Simple dot product with custom embedding layer --------------

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
    
    self$dot <-
      layer_lambda(
        output_shape = self$embedding_dim,
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

model %>% compile(
  loss = "mse",
  optimizer = "adam"
)

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)



