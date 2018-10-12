# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb

library(keras)
use_implementation("tensorflow")
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)

images %<-% tf$keras$datasets$mnist$load_data()[[1]][[1]]
images %>% dim()

images[images[, , ] < 127] <- 0
images[images[, , ] >= 127] <- 1


images <- images %>% k_reshape(c(60000, 28, 28, 1))

buffer_size <- 1000
batch_size <- 100

dataset <- tensor_slices_dataset(images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)


latent_dim <- 50

encoder <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      layer_conv_2d(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$conv1 <-
      layer_conv_2d(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = 2 * latent_dim)
    
    function (x, mask = NULL) {
      x %>% 
        self$conv1() %>%
        self$conv2() %>%
        self$flatten() %>% 
        self$dense() %>%
        tf$split(num_or_size_splits = 2, axis = 1)
    }
  })
}

decoder <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$dense <- layer_dense(units = 7 * 7 * 32, activation = "relu")
    self$reshape <- layer_reshape(target_shape = c(7, 7, 32))
    
    self$deconv1 <-
      layer_conv_2d_transpose(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2 <-
      layer_conv_2d(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv3 <-
      layer_conv_2d_transpose(
        filters = 1,
        kernel_size = 3,
        strides = 1,
        padding = "same"
      )

    
    function (x, mask = NULL) {
      x %>% 
        self$dense() %>% 
        self$reshape() %>%
        self$deconv1() %>% 
        self$deconv2() %>% 
        self$deconv3()
      
      ##if apply_sigmoid:       probs = tf.sigmoid(logits)
    }
  })
}


# Sampling and loss --------------------------------------------------------------------

reparameterize <- function(mean, logvar) {
  eps = tf$random_normal(shape = mean$shape)
  tf$exp(logvar * .5) + mean
}

# In practice, we optimize the single sample Monte Carlo estimate of this expectation:
# log p(x| z) + log p(z) - log q(z|x)
# where z is sampled from q(z|x).
log_normal_pdf <- function(sample, mean, logvar, reduce_axis = 1) {
  log2pi = tf$log(2 * np$pi)
  tf$reduce_sum(
    -.5 * ((sample - mean)^2 * tf$exp(-logvar) + logvar + log2pi),
    axis = reduce_axis)
}
 
optimizer <- tf$train$AdamOptimizer(1e-4)


# Generation --------------------------------------------------------------

random_vector_for_generation <- tf$random_normal(
  shape = list(num_examples_to_generate, latent_dim))

generate_and_save_images <- function(
  epoch) {
  predictions <-decoder(random_vector_for_generation) %>% tf$nn$sigmoid()
  # tbd plot
}



# Training loop -----------------------------------------------------------

num_epochs <- 10#0


for (epoch in seq_len(num_epochs)) {
  iter <- make_iterator_one_shot(dataset)
  total_loss <- 0
  
  until_out_of_range({
    x <-  iterator_get_next(iter)
    
    with(tf$GradientTape() %as% tape, {
      
      c(mean, logvar) %<-% encoder(x)
      z <- reparameterize(mean, logvar)
      preds <- decoder(z)
      crossentropy_loss <-
        tf$nn$sigmoid_cross_entropy_with_logits(logits = preds, labels = x)
      logpx_z <- -tf$reduce_sum(crossentropy_loss, axis = list(1, 2, 3))
      logpz <- log_normal_pdf(z, 0, 0)
      logqz_x <- log_normal_pdf(z, mean, logvar)
      loss <- -tf$reduce_mean(logpx_z + logpz - logqz_x)
      
    })
    
    total_loss <- total_loss + loss

    gradients <- tape$gradient(loss, model$variables)
    optimizer$apply_gradients(purrr::transpose(list(gradients, model$variables)),
                              global_step = tf$train$get_or_create_global_step())
    
  })
  
  cat("Total loss (epoch): ", i, ": ", as.numeric(total_loss), "\n")
  
  if (epoch %% 1 == 0) {
    generate_and_save_images(
      epoch, random_vector_for_generation)
  }
}
