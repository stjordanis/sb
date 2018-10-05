
source("config.R")
source("preproc.R")
source("utils.R")


# Simple object detection (one anchor box only) -------------------------------------------------

# what object is located in each cell of a 4x4 grid

size <- 224

n_anchors_per_cell <- 1
n_classes <- 21

n_filters <- n_anchors_per_cell * (4 + n_classes)


# construct anchors 

gridsize <- 4

anchor_offset <- 1/(gridsize*2)

anchor_xs <- seq(anchor_offset, 1 - anchor_offset, length.out = 4) %>% rep(each = gridsize)
anchor_ys <- seq(anchor_offset, 1 - anchor_offset, length.out = 4) %>% rep(gridsize)

anchor_centers <- cbind(anchor_xs, anchor_ys) %>% k_constant()
anchor_height_width <- matrix(1/gridsize, nrow = 16, ncol = 2) %>% k_constant()
anchors <- k_concatenate(list(anchor_centers, anchor_height_width))
ggplot(data.frame(x = anchor_xs, y = anchor_ys), aes(x, y)) + geom_point() + theme(aspect.ratio = 1)

height_width2corners <- function(centers, height_width) {
  k_concatenate(list(centers - height_width/2, centers + height_width/2))
}
anchor_corners <- height_width2corners(anchor_centers, anchor_height_width)

# ssd loss
# incoming shapes
# torch.Size([16, 21])
# torch.Size([16, 4])
# torch.Size([56])
# torch.Size([14])

ssd_loss <- function(y_true, y_pred) {
  
  loc_loss_batch <- k_constant(0)
  class_loss_batch <- k_constant(0)
  
  for (i in 1:nrow(y_true)) {
    y_true_bbox <- y_true[i, 1:12]
    y_true_class <- y_true[i, 13:15]
    # size (16, 4)
    y_pred_bbox <- y_pred[i, , 1:4]
    # size (16, 21)
    y_pred_class <- y_pred[i, , 5:25]
    
    losses <- ssd_loss_single(y_true_bbox, y_true_class, y_pred_bbox, y_pred_class)
    loc_loss_batch <- loc_loss_batch + losses[[1]]
    class_loss_batch <- class_loss_batch + losses[[2]]
  }
  #list(loc_loss_batch, class_loss_batch)
  k_constant(1L)
}

activations_to_bboxes <- function(activations, anchors) {
  activations <- k_tanh(activations)
  activation_centers <- activations[ , 1:2]/2 * gridsize + anchors[ , 1:2]
  activation_height_width <- activations[ , 3:4]/(2 + 1) + anchors[ , 3:4]
  activation_corners <- height_width2corners(activation_centers, activation_height_width)
  activation_corners
}

get_y <- function(true_bbox,true_class) {
  true_bbox <- k_reshape(true_bbox, c(-1, 4))
  #print(true_bbox %>% k_eval())
  true_bbox <- true_bbox/size
  mask <- tf$greater(true_bbox[ , 3] - true_bbox[ , 1], 0)
  true_bbox_keep <- tf$boolean_mask(true_bbox, mask)
  true_class_keep <- tf$boolean_mask(true_class, mask) %>% k_reshape(c(-1, 1))
  #print(true_bbox_keep %>% k_eval())
  #print(true_class_keep %>% k_eval())
  concat <- k_concatenate(list(true_bbox_keep, true_class_keep))
  # print(concat %>% k_eval())
  concat
}

intersect <- function(box1, box2) {
  #print(box1 %>% k_eval())
  #print(box2 %>% k_eval())
  #print(k_shape(box1)%>% k_eval())
  #print(k_shape(box2)%>% k_eval())
  
  box1_a <- box1[ , 3:4] %>% k_expand_dims(axis = 2)
  box2_a <- box2[ , 3:4] %>% k_expand_dims(axis = 1)
  #print(box1_a %>% k_eval())
  #print(box2_a %>% k_eval())
  #print(k_shape(box1_a)%>% k_eval())
  #print(k_shape(box2_a)%>% k_eval())
  max_xy <- k_minimum(box1_a, box2_a)
  #print(max_xy %>% k_eval())
  #print(k_shape(max_xy) %>% k_eval())
  
  box1_b <- box1[ , 1:2] %>% k_expand_dims(axis = 2)
  box2_b <- box2[ , 1:2] %>% k_expand_dims(axis = 1)
  #print(box1_b %>% k_eval())
  #print(box2_b %>% k_eval())
  #print(k_shape(box1_b)%>% k_eval())
  #print(k_shape(box2_b)%>% k_eval())
  min_xy <- k_maximum(box1_b, box2_b)
  #print(min_xy %>% k_eval())
  #print(k_shape(min_xy) %>% k_eval())
  intersection <- k_clip(max_xy - min_xy, min = 0, max = Inf)
  intersection[ , , 1] * intersection[, , 2]

}

box_size <- function(box) {
  (box[ , 3] - box[ , 1]) * (box[ , 4] - box[ , 2])
} 

jaccard <- function(bbox, anchor_corners) {
  intersection <- intersect(bbox, anchor_corners)
  #print(intersection %>% k_eval())
  union <- k_expand_dims(box_size(bbox), axis = 2)  + k_expand_dims(box_size(anchor_corners), axis = 1) - intersection
  #print(union %>% k_eval())
  intersection/union
}

map_to_ground_truth <- function(overlaps) {
  #print(overlaps %>% k_eval())
  prior_overlap <- k_max(overlaps, axis = 2)
  #print(prior_overlap %>% k_eval())
  prior_idx <- k_argmax(overlaps, axis = 2)
  #print(prior_idx %>% k_eval())
  gt_overlap <- k_max(overlaps, axis = 1)
  gt_idx <- k_argmax(overlaps, axis = 1)
  
  gt_overlap_eval <- gt_overlap %>% k_eval()
  prior_idx_eval <- prior_idx %>% k_eval()

  prior_idx_eval <- prior_idx_eval + 1 ##### !!
  gt_idx_eval <- gt_idx %>% k_eval()
  gt_idx_eval <- gt_idx_eval + 1 ##### !!

  gt_overlap_eval[prior_idx_eval] <- 1.99

  for(i in 1:length(prior_idx_eval)) {
    p <- prior_idx_eval[i]
    gt_idx_eval[p] <- i
  }
  
  #print(gt_overlap_eval)
  #print(gt_idx_eval) 
  k_concatenate(list(k_constant(gt_overlap_eval), k_constant(gt_idx_eval - 1))) # subtracting again

}

ssd_loss_single <- function(y_true_bbox, y_true_class, y_pred_bbox, y_pred_class) {
  acts <- activations_to_bboxes(y_pred_bbox, anchors)
  y_extracted <- get_y(y_true_bbox, y_true_class)
  y_true_bbox <- y_extracted[ , 1:4]
  y_true_class <- y_extracted[ , 5]
  
  overlaps <- jaccard(y_true_bbox, anchor_corners)
  #print(overlaps %>% k_eval())
  overlaps <- map_to_ground_truth(overlaps)
  #print(overlaps)
  gt_overlap <- overlaps[1:16]
  gt_idx <- overlaps[17:32] %>% k_cast("int32")
  #print(gt_overlap %>% k_eval())
  
  #print(y_true_class %>% k_eval())
  #print(gt_idx %>% k_eval())
  
  gt_class <- k_gather(y_true_class, gt_idx)
  print(gt_class %>% k_eval())
  
  
  list(1L, 1L)
}
  
#parent_set <- k_constant(1:6)
#new_set <- k_eval(parent_set)
#indices <- k_constant(c(1L,4L), dtype = "int32")
#indices_vals <- k_eval(indices)
#new_set[indices_vals] <- 1.99
#new_set

#import keras
#import keras.backend as k
#import numpy as np
#parent_set = k.constant(np.arange(6))
#indices = k.constant([1,4], dtype="int32")
#parent_set[2] = 1.99


# pos = gt_overlap > 0.4
# pos_idx = torch.nonzero(pos)[:,0]
# gt_clas[1-pos] = len(id2cat)
# gt_bbox = bbox[gt_idx]
# loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
# clas_loss  = loss_f(b_c, gt_clas)

ssd_loss(test_y_true, test_y_pred) %>% k_eval()

test_y_pred <- array(runif(batch_size * 16 * 25), dim = c(batch_size, 16, 25)) %>% k_constant()

test_y_true_cl <- matrix(c(1,14,14,0,11,17), nrow = 2, ncol = 3, byrow = TRUE)
test_y_true_bb <- matrix(c(91,47,223,169,0,49,205,180,9,169,217,222,
                           0,0,0,0,81,49,166,171,63,1,222,223), nrow = 2, ncol = 12, byrow = TRUE)
test_y_true <- cbind(test_y_true_bb, test_y_true_cl) %>% k_constant()

#ccc <- k_constant(c(1,14,14))
#ddd <- k_constant(c(0L ,1L, 1L, 1L, 2L, 1L, 1L, 2L, 2L, 2L, 1L, 2L, 2L, 2L, 2L, 2L), dtype = "int32")
#k_gather(ccc,ddd) %>% k_eval()

feature_extractor <- application_resnet50(include_top = FALSE, input_shape = c(224, 224, 3))

feature_extractor

input <- feature_extractor$input
common <- feature_extractor$output %>%
  layer_conv_2d(filters = 256, kernel_size = 3, padding = "same", name = "head_conv1") %>%
  layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, padding = "same", name = "head_conv2")

bbox_conv <- layer_conv_2d(common, filters = 4, kernel_size = 3, padding = "same", name = "bbox_conv") %>%
  layer_reshape(target_shape = c(-1, 4), name = "bbox_flatten")
class_conv <- layer_conv_2d(common, filters = n_classes, kernel_size = 3, padding = "same", name = "class_conv") %>%
  layer_reshape(target_shape = c(-1, 21), name = "class_flatten")
concat <- layer_concatenate(list(bbox_conv, class_conv))

model <-
  keras_model(inputs = input,
              outputs = concat)

# outputs should be ([bs, 16, 4]) and ([bs, 16, 21])
# total output is (bs, 16, 25)

# a = torch.rand(64, 4, 4, 21)
# a.shape
# a.view(64, -1, 21).shape

model %>% freeze_weights()
model
model %>% unfreeze_weights(from = "head_conv1")
model
       