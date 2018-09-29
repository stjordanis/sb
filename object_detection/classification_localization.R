source("preproc.R")

# use only largest object per image
imageinfo_maxbb <- imageinfo %>%
  group_by(id) %>% 
  filter(which.max(area)==row_number())
nrow(imageinfo_maxbb)


## classification of largest object

model <- application_resnet50()

img_array <- image_load(file.path(img_dir, imageinfo_maxbb$file_name[[1]]), target_size = c(224, 224)) %>%
  image_to_array() %>%
  imagenet_preprocess_input() 
dim(img_array) <- c(1, dim(img_array))

model %>% predict(img_array, batch_size = 1) %>% imagenet_decode_predictions()


## localization of largest object

feature_extractor <- application_resnet50(include_top = FALSE, pooling = "avg")

feature_extractor %>% freeze_weights()
  
model <- keras_model_sequential() %>%
  feature_extractor %>%
  layer_dense(units = 32) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 4)
model


model %>% compile(optimizer = "adam", loss = "mae", metrics = "mae")

localization_generator <- function(data, shuffle, batch_size = 2) {
 
  i <- 1
  function() {
    if (shuffle) {
      indices <- sample(1:nrow(data), size = batch_size)
    } else {
      if (i + batch_size >= nrow(data)) i <<- 1
      indices <- c(i:min(i + batch_size - 1, nrow(data)))
      i <<- i + length(indices)
    }
    
    # samples <- array(0, dim = c(length(rows), 
    #                             lookback / step,
    #                             dim(data)[[-1]]))
    # targets <- array(0, dim = c(length(rows)))
    # 
    # for (j in 1:length(rows)) {
    #   indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, 
    #                  length.out = dim(samples)[[2]])
    #   samples[j,,] <- data[indices,]
    #   targets[[j]] <- data[rows[[j]] + delay,2]
    # }            
    # 
    # list(samples, targets)
  }
}

train_gen <- localization_generator(
  imageinfo_maxbb
)