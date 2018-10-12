library(keras)
# make sure we use the tensorflow implementation of Keras
# this line has to be executed immediately after loading the library
use_implementation("tensorflow")

library(tensorflow)
# enable eager execution
# the argument device_policy is needed only when using a GPU
tfe_enable_eager_execution(device_policy = "silent")

iris_regression_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    feature_extractor <- application_resnet50(include_top = FALSE, input_shape = c(224, 224, 3))
    
    # this is the "call" function that defines what happens when the model is called
    function (x, mask = NULL) {
      x %>% 
        feature_extractor()
    }
  })
}

model <- iris_regression_model()

t <- array(runif(10 * 224 * 224 * 3), dim = c(10, 224, 224, 3)) %>% k_constant()
model(t)
