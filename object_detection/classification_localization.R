


source("config.R")
source("preproc.R")
source("utils.R")

# use only largest object per image
imageinfo_maxbb <- imageinfo %>%
  group_by(id) %>%
  filter(which.max(area) == row_number())

n_samples <- nrow(imageinfo_maxbb)

# train/test split/
train_indices <- sample(1:n_samples, 0.8 * n_samples)
train_data <- imageinfo_maxbb[train_indices,]
validation_data <- imageinfo_maxbb[-train_indices,]



# classification of largest object ----------------------------------------

# epoch 11: accs 0.68/0.74

feature_extractor <-
  application_resnet50(
    include_top = FALSE,
    input_shape = c(224, 224, 3),
    pooling = "avg"
  )

feature_extractor

feature_extractor %>% freeze_weights()

model <- keras_model_sequential() %>%
  feature_extractor %>%
  ### absolutely required!!!
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 20, activation = "softmax")

model

model %>% compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = list("accuracy"))

classification_generator <-
  function(data,
           target_height,
           target_width,
           shuffle,
           batch_size) {
    i <- 1
    function() {
      if (shuffle) {
        indices <- sample(1:nrow(data), size = batch_size)
      } else {
        if (i + batch_size >= nrow(data))
          i <<- 1
        indices <- c(i:min(i + batch_size - 1, nrow(data)))
        i <<- i + length(indices)
      }
      x <-
        array(0, dim = c(length(indices), target_height, target_width, 3))
      y <- array(0, dim = c(length(indices), 1))
      
      for (j in 1:length(indices)) {
        x[j, , ,] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        y[j,] <-
          data[[indices[j], "category_id"]] - 1
      }
      list(x, y)
    }
  }

train_gen <- classification_generator(
  train_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = TRUE,
  batch_size = batch_size
)

valid_gen <- classification_generator(
  validation_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = FALSE,
  batch_size = batch_size
)

model %>% fit_generator(
  train_gen,
  epochs = 50,
  steps_per_epoch = nrow(train_data) / batch_size,
  validation_data = valid_gen,
  validation_steps = nrow(validation_data) / batch_size,
  callbacks = list(
    callback_model_checkpoint(
      file.path("class_only", "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ),
    callback_early_stopping(patience = 10)
  )
)

batch <- valid_gen()

preds <- model %>% predict(batch[[1]], batch_size = batch_size)

# tbd decode


#  localization of largest object -----------------------------------------

# 38 70

feature_extractor <-
  application_resnet50(
    include_top = FALSE,
    input_shape = c(224, 224, 3)
  )

feature_extractor

feature_extractor %>% freeze_weights()

model <- keras_model_sequential() %>%
  feature_extractor %>%
  layer_flatten() %>%
 # layer_dropout(rate = 0.25) %>%
  layer_dense(units = 4)

model

model %>% compile(
  optimizer = "adam",
  loss = "mae",
  metrics = list(custom_metric("iou", metric_iou))
)

localization_generator <-
  function(data,
           target_height,
           target_width,
           shuffle,
           batch_size) {
    i <- 1
    function() {
      if (shuffle) {
        indices <- sample(1:nrow(data), size = batch_size)
      } else {
        if (i + batch_size >= nrow(data))
          i <<- 1
        indices <- c(i:min(i + batch_size - 1, nrow(data)))
        i <<- i + length(indices)
      }
      x <-
        array(0, dim = c(length(indices), target_height, target_width, 3))
      y <- array(0, dim = c(length(indices), 4))
      
      for (j in 1:length(indices)) {
        x[j, , ,] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        y[j,] <-
          data[indices[j], c("x_left", "y_top", "x_right", "y_bottom")] %>% as.matrix()
      }
      list(x, y)
    }
  }

train_gen <- localization_generator(
  train_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = TRUE,
  batch_size = batch_size
)

valid_gen <- localization_generator(
  validation_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = FALSE,
  batch_size = batch_size
)

model %>% fit_generator(
  train_gen,
  epochs = 50,
  steps_per_epoch = nrow(train_data) / batch_size,
  validation_data = valid_gen,
  validation_steps = nrow(validation_data) / batch_size,
  callbacks = list(
    callback_model_checkpoint(
      file.path("loc_only", "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ),
    callback_early_stopping(patience = 10)
  )
)

batch <- valid_gen()

preds <- model %>% predict(batch[[1]], batch_size = batch_size)

i <- 3
plot_image_with_boxes(validation_data[i, ], box_pred = preds[i, ])


#  classification plus localization of largest object ---------------------

feature_extractor <-
  application_resnet50(
    include_top = FALSE,
    input_shape = c(224, 224, 3)
  )

feature_extractor

input <- feature_extractor$input
common <- feature_extractor$output %>%
  layer_flatten(name = "flatten") %>%
  layer_activation_relu() %>%
  #layer_dropout(rate = 0.5) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_batch_normalization() #%>%
 # layer_dropout(rate = 0.5) 

regression_output <- layer_dense(common, units = 4, name = "regression_output")
class_output <-
  layer_dense(common, units = 20, activation = "softmax", name = "class_output")

model <-
  keras_model(inputs = input,
              outputs = list(regression_output, class_output))

model %>% freeze_weights(to = "flatten")
model
model %>% compile(
  optimizer = "adam",
  loss = list("mae", "sparse_categorical_crossentropy"),
  loss_weights = list(regression_output = 0.05, class_output = 0.95),
  metrics = list(
    regression_output = custom_metric("iou", metric_iou),
    class_output = "accuracy"
  )
)

loc_class_generator <-
  function(data,
           target_height,
           target_width,
           shuffle,
           batch_size) {
    i <- 1
    function() {
      if (shuffle) {
        indices <- sample(1:nrow(data), size = batch_size)
      } else {
        if (i + batch_size >= nrow(data))
          i <<- 1
        indices <- c(i:min(i + batch_size - 1, nrow(data)))
        i <<- i + length(indices)
      }
      x <-
        array(0, dim = c(length(indices), target_height, target_width, 3))
      y1 <- array(0, dim = c(length(indices), 4))
      y2 <- array(0, dim = c(length(indices), 1))
      
      for (j in 1:length(indices)) {
        x[j, , ,] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        y1[j,] <-
          data[indices[j], c("x_left", "y_top", "x_right", "y_bottom")] %>% as.matrix()
        y2[j,] <-
          data[[indices[j], "category_id"]] - 1
      }
      list(x, list(y1, y2))
    }
  }

train_gen <- loc_class_generator(
  train_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = TRUE,
  batch_size = batch_size
)

valid_gen <- loc_class_generator(
  validation_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = FALSE,
  batch_size = batch_size
)

model %>% fit_generator(
  train_gen,
  epochs = 50,
  steps_per_epoch = nrow(train_data) / batch_size,
  validation_data = valid_gen,
  validation_steps = nrow(validation_data) / batch_size,
  callbacks = list(
    callback_model_checkpoint(
      file.path("loc_class", "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ),
    callback_early_stopping(patience = 10)
  )
)

# multi-object classification ---------------------------------------------

image_cats <- imageinfo %>% select(category_id) %>% 
  mutate(category_id = category_id - 1) %>% pull() %>%
  to_categorical(num_classes = 20)

image_cats <- data.frame(image_cats) %>% add_column(file_name = imageinfo$file_name, .before = TRUE) 

image_cats <- image_cats %>% group_by(file_name) %>% summarise_all(.funs = funs(max))

n_samples <- nrow(image_cats)

# train/test split
train_indices <- sample(1:n_samples, 0.8 * n_samples)
train_data <- image_cats[train_indices, ]
validation_data <- image_cats[-train_indices, ]

feature_extractor <-
  application_resnet50(
    include_top = FALSE,
    input_shape = c(224, 224, 3),
    pooling = "avg"
  )

feature_extractor

feature_extractor %>% freeze_weights()

model <- keras_model_sequential() %>%
  feature_extractor %>%
  ### absolutely required!!!
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 20, activation = "sigmoid")

model

model %>% compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics = list("accuracy"))

classification_generator <-
  function(data,
           target_height,
           target_width,
           shuffle,
           batch_size) {
    i <- 1
    function() {
      if (shuffle) {
        indices <- sample(1:nrow(data), size = batch_size)
      } else {
        if (i + batch_size >= nrow(data))
          i <<- 1
        indices <- c(i:min(i + batch_size - 1, nrow(data)))
        i <<- i + length(indices)
      }
      x <-
        array(0, dim = c(length(indices), target_height, target_width, 3))
      y <- array(0, dim = c(length(indices), 20))
      
      for (j in 1:length(indices)) {
        x[j, , ,] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        y[j,] <-
          data[indices[j], 2:21] %>% as.matrix()
      }
      list(x, y)
    }
  }

train_gen <- classification_generator(
  train_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = TRUE,
  batch_size = batch_size
)

valid_gen <- classification_generator(
  validation_data,
  target_height = target_height,
  target_width = target_width,
  shuffle = FALSE,
  batch_size = batch_size
)

model %>% fit_generator(
  train_gen,
  epochs = 50,
  steps_per_epoch = nrow(train_data) / batch_size,
  validation_data = valid_gen,
  validation_steps = nrow(validation_data) / batch_size,
  callbacks = list(
    callback_model_checkpoint(
      file.path("multiclass", "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ),
    callback_early_stopping(patience = 10)
  )
)




