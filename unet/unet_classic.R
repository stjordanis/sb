# https://github.com/SimonKohl/probabilistic_unet/blob/master/model/probabilistic_unet.py

reticulate::use_condaenv("tf-12-gpu-0924", required = TRUE)

library(keras)

down_block <- function(
  name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    self$conv1 <- layer_conv_2d(filters = , 
                           kernel_size = ,
                           strides = ,
                           )

    function(inputs, mask = NULL) {
      
      
    }
  })
}


# features = [inputs]
# 
# # iterate blocks (`processing scales')
# for i, n_channels in enumerate(self._num_channels):
#   
#   if i == 0:
#       down_sample = False
#   else:
#     down_sample = True
#   tf.logging.info('encoder scale {}: {}'.format(i, features[-1].get_shape()))
#   features.append(down_block(features[-1],
#                            output_channels=n_channels,
#                            kernel_shape=(3,3),
#                            num_convs=self._num_convs,
#                            nonlinearity=self._nonlinearity,
#                            initializers=self._initializers,
#                            regularizers=self._regularizers,
#                            down_sample_input=down_sample,
#                            data_format=self._data_format,
#                            down_sampling_op=self._down_sampling_op,
#                            name='down_block_{}'.format(i)))
# # return all features except for the input images
# return features[1:]

vgg_encoder <- function(num_channels = num_channels,
                        num_convs_per_block = num_convs_per_block,
                        down_sampling_op = down_sampling_op) {
  
  keras_model_custom(name = name, function(self) {
    
    down1 <- down_block(
    )
    
    function(inputs, mask = NULL) {
      
    }
    
  })
}

vgg_decoder <- function(num_channels = num_channels,
                        num_classes = num_classes,
                        num_convs_per_block = num_convs_per_block,
                        up_sampling_op = up_sampling_op) {
  
  keras_model_custom(name = name, function(self) {
    
    function(inputs, mask = NULL) {
      
    }
    
  })
}
  


unet <- function(num_channels = num_channels,
                 num_classes = num_classes,
                 num_convs_per_block = num_convs_per_block,
                 down_sampling_op = down_sampling_op,
                 up_sampling_op = up_sampling_op,
                 name = name) {
  
  keras_model_custom(name = name, function(self) {
    
    self$encoder <- vgg_encoder(num_channels,
                                num_convs_per_block,
                                down_sampling_op)
    self$decoder <- vgg_decoder(num_channels,
                                num_classes,
                                num_convs_per_block,
                                up_sampling_op)
    
    function(inputs, mask = NULL) {
      
      encoder_features <- self$encoder(inputs)
      predicted_logits <- self$decoder(encoder_features)
      predicted_logits
      
    }
  })
}





########################################################



base_channels <- 32
num_channels <- list(base_channels, 2 * base_channels, 4 * base_channels,
                 6 * base_channels, 6 * base_channels, 6 * base_channels, 6 * base_channels)
num_classes <- 19
num_convs_per_block <- 3

down_sampling_op <- function(x, df) {
  tf$nn$avg_pool(x,
                 ksize = list(1,1,2,2),
                 strides = list(1,1,2,2),
                 padding = 'SAME')
}
  
up_sampling_op <- function(x, size) {
 tf$image$resize_images(x, size, method = tf$image$ResizeMethod.BILINEAR, align_corners = True)
}



unet <- unet(
  num_channels = num_channels,
  num_classes = num_classes,
  num_convs_per_block = num_convs_per_block,
  nonlinearity = nonlinearity,
  initializers = initializers,
  regularizers = regularizers,
  down_sampling_op = down_sampling_op,
  up_sampling_op = up_sampling_op,
  name = "unet"
  )
  
  
  
  
                     
