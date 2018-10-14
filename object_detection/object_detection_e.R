library(keras)
use_implementation("tensorflow")

library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")


n_classes <- 20

detection_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    feature_extractor <- application_resnet50(include_top = FALSE, input_shape = c(224, 224, 3), pooling = NULL)
    conv1 <- layer_conv_2d(filters = 256, kernel_size = 3, padding = "same", name = "head_conv1")
    conv2 <- layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, padding = "same", name = "head_conv2")
    bbox_conv <- layer_conv_2d(filters = 4, kernel_size = 3, padding = "same", name = "bbox_conv") 
    bbox_reshape <- layer_reshape(target_shape = c(-1, 4), name = "bbox_flatten")
    class_conv <- layer_conv_2d(filters = n_classes + 1, kernel_size = 3, padding = "same", name = "class_conv") 
    class_reshape <- layer_reshape(target_shape = c(-1, 21), name = "class_flatten")

    function (x, mask = NULL) {
      common <- x %>% 
        feature_extractor() %>%
        conv1() %>%
        conv2()
      print(common %>% dim())
      bbox_out <- common %>% 
        bbox_conv() %>%
        bbox_reshape()
      print(bbox_out %>% dim())
      class_out <- common %>%
        class_conv() %>%
        class_reshape()
      print(class_out %>% dim())
      k_concatenate(list(bbox_out, class_out))
    }
  })
}

model <- detection_model()
# freeze

t <- array(runif(10 * 224 * 224 * 3), dim = c(10, 224, 224, 3)) %>% k_constant()
model(t) %>% dim()

