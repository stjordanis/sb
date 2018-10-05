reticulate::use_condaenv("tf-12-gpu-0924")

library(keras)
library(tensorflow)
library(rjson)
library(magick)
library(purrr)
library(tibble)
library(tidyr)
library(dplyr)
library(ggplot2)

target_height <- 224
target_width <- 224

batch_size <- 4
