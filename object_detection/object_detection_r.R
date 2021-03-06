
source("config.R")
source("preproc.R")
source("utils.R")


# Simple object detection (one anchor box per grid cell) -------------------------------------------------

# this will find up to one object per grid cell 



# Prepare data ------------------------------------------------------------

imageinfo4ssd <- imageinfo %>%
  select(category_id,
         file_name,
         ends_with("scaled"))


imageinfo4ssd <- imageinfo4ssd %>% 
  group_by(file_name) %>% 
  summarise(categories = toString(category_id),
            xl = toString(x_left_scaled),
            yt = toString(y_top_scaled),
            xr = toString(x_right_scaled),
            yb = toString(y_bottom_scaled),
            cnt = n())

# Construct anchors  ------------------------------------------------------

# actual number of classes (without background)
n_classes <- 20
class_background <- 21

# construct a 4x4 grid
gridsize <- 4
anchor_offset <- 1/(gridsize*2) # 0.25

# x resp. y coordinates of grid centers 
anchor_xs <- seq(anchor_offset, 1 - anchor_offset, length.out = 4) %>% rep(each = gridsize)
anchor_ys <- seq(anchor_offset, 1 - anchor_offset, length.out = 4) %>% rep(gridsize)

anchor_centers <- cbind(anchor_xs, anchor_ys) 
anchor_height_width <- matrix(1/gridsize, nrow = 16, ncol = 2) 
# anchor information is x/y coords of center, height, width
anchors <- cbind(anchor_centers, anchor_height_width)
ggplot(data.frame(x = anchor_xs, y = anchor_ys), aes(x, y)) + geom_point() + theme(aspect.ratio = 1)

# convert from center-height-width representation to grid corners
hw2corners <- function(centers, height_width) {
  cbind(centers - height_width/2, centers + height_width/2) %>% unname()
}
anchor_corners <- hw2corners(anchor_centers, anchor_height_width)



# Matching ----------------------------------------------------------------

# compute IOU
jaccard <- function(bbox, anchor_corners) {
  bbox <- k_constant(bbox)
  anchor_corners <- k_constant(anchor_corners)
  intersection <- intersect(bbox, anchor_corners)
  union <- k_expand_dims(box_area(bbox), axis = 2)  + k_expand_dims(box_area(anchor_corners), axis = 1) - intersection
  res <- intersection/union
  res %>% k_eval()
}

# compute intersection for IOU
intersect <- function(box1, box2) {
  
  box1_a <- box1[ , 3:4] %>% k_expand_dims(axis = 2)
  box2_a <- box2[ , 3:4] %>% k_expand_dims(axis = 1)
  max_xy <- k_minimum(box1_a, box2_a)
  
  box1_b <- box1[ , 1:2] %>% k_expand_dims(axis = 2)
  box2_b <- box2[ , 1:2] %>% k_expand_dims(axis = 1)
  min_xy <- k_maximum(box1_b, box2_b)
  
  intersection <- k_clip(max_xy - min_xy, min = 0, max = Inf)
  intersection[ , , 1] * intersection[, , 2]
  
}

box_area <- function(box) {
  (box[ , 3] - box[ , 1]) * (box[ , 4] - box[ , 2]) ## these are corners!
} 


# determine best overlap of ground truth boxes and anchor (prior) boxes 
map_to_ground_truth <- function(overlaps) {
  
  # overlaps shape is: number of ground truth objects * number of grid cells
  # for each ground truth object, find maximally overlapping cell (crit. 1)
  # shape: number of ground truth objects
  prior_overlap <- apply(overlaps, 1, max)
  prior_idx <- apply(overlaps, 1, which.max)
  
  # for each grid cell, what object does it overlap with most (crit. 2)
  # shape: number of grid cells
  gt_overlap <-  apply(overlaps, 2, max)
  gt_idx <- apply(overlaps, 2, which.max)
  
  # set all definitely overlapping cells to respective object (crit. 1)
  gt_overlap[prior_idx] <- 1.99

  # now still set all others to best match by crit. 2
  for(i in 1:length(prior_idx)) {
    p <- prior_idx[i]
    gt_idx[p] <- i
  }
  
  # return: vectors of length (number of grid cells), containing measures of overlap as well as object indices
  list(gt_overlap, gt_idx)
  
}


# Generator ---------------------------------------------------------------

image_size <- target_height 

ssd_generator <-
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
      y1 <- array(0, dim = c(length(indices), 16))
      y2 <- array(0, dim = c(length(indices), 16, 4))
      
      for (j in 1:length(indices)) {
        print(j)
        x[j, , , ] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        
        class_string <- data[indices[j], ]$categories
        xl_string <- data[indices[j], ]$xl
        yt_string <- data[indices[j], ]$yt
        xr_string <- data[indices[j], ]$xr
        yb_string <- data[indices[j], ]$yb
        
        classes <-  str_split(class_string, pattern = ", ")[[1]]
        xl <-  str_split(xl_string, pattern = ", ")[[1]] %>% as.double() %>% `/`(image_size)
        yt <-  str_split(yt_string, pattern = ", ")[[1]] %>% as.double() %>% `/`(image_size)
        xr <-  str_split(xr_string, pattern = ", ")[[1]] %>% as.double() %>% `/`(image_size)
        yb <-  str_split(yb_string, pattern = ", ")[[1]] %>% as.double() %>% `/`(image_size)
        boxes <- rbind(xl, yt, xr, yb)
       
        bbox <- cbind(xl, yt, xr, yb)
        overlaps <- jaccard(bbox, anchor_corners)
        c(gt_overlap, gt_idx) %<-% map_to_ground_truth(overlaps)
        gt_class <- classes[gt_idx]
        pos <- gt_overlap > 0.4
        gt_class[gt_overlap < 0.4] <- 21
        gt_bbox <- boxes[ , gt_idx]
        gt_bbox[ , !pos] <- 0
        gt_bbox <- gt_bbox %>% t()
        y1[j, ] <- gt_class
        y2[j, , ] <- gt_bbox

      }
      x <- x/255
      list(x, list(y1, y2))
    }
  }




train_gen <- ssd_generator(
  imageinfo4ssd,
  target_height = target_height,
  target_width = target_width,
  #shuffle = TRUE,
  shuffle = FALSE,
  batch_size = batch_size
)

batch <- train_gen()
c(x,c(y1, y2)) %<-% batch
x %>% dim()
y1
y2[1, , ]
y2[3, , ]

# SSD loss ----------------------------------------------------------------

### overall loss
ssd_loss <- function(y_true, y_pred) {
  
  loc_loss_batch <- k_constant(0)
  class_loss_batch <- k_constant(0)
  
  batch_shape <- k_shape(y_true)[1]
  i <- k_constant(1L, dtype = "int32")
  max_pad <- k_constant(as.integer(max_pad), dtype = "int32")
  classes_start <- max_pad * 4L + 1L
  last_pos <- max_pad * 4L + max_pad
  
  condition <- function(
    y_true, y_pred, loc_loss_batch, class_loss_batch, i, batch_shape) tf$less(i, batch_shape)
  
  do <- function(y_true, y_pred, loc_loss_batch, class_loss_batch, i, batch_shape) {
    #y_true_bbox <- y_true[i, 1L:k_cast(max_pad * 4L, "int32")]
    y_true_class <- y_true[i, classes_start:last_pos]
    y_pred_bbox <- y_pred[i, , 1L:4L] 
    y_pred_class <- y_pred[i, , 5L:25L]
    list(
      y_true, y_pred, loc_loss_batch, class_loss_batch, tf$add(i, 1L), batch_shape
    )
  }

  tf$while_loop(
    condition,
    do,
    list(y_true, y_pred, loc_loss_batch, class_loss_batch, i, batch_shape)
  )
 
  loc_loss_batch + class_loss_batch
  

}

### loss per image
ssd_loss_single <- function(y_true_bbox, y_true_class, y_pred_bbox, y_pred_class) {
  
  # convert activations into bounding boxes
  acts <- activations_to_bboxes(y_pred_bbox, anchors)
  
  #print(acts %>% k_eval())
  # calculate localization loss for all boxes with sufficient overlap
  loc_loss <- (tf$gather(acts, pos_idx) - tf$gather(gt_bbox, pos_idx)) %>% tf$abs() %>% tf$reduce_mean()
  #print(loc_loss %>% k_eval())
  
  print(y_pred_class)
  # one-hot-encode ground truth labels
  gt_class <- tf$one_hot(gt_class, depth = class_background)
  #print(gt_class)
  
  # leave out background for class loss calculation
  class_loss  <- tf$nn$sigmoid_cross_entropy_with_logits(labels = gt_class[ , 1:20], logits = y_pred_class[ , 1:20])
  # divide by number of classes
  class_loss <- tf$reduce_sum(class_loss)/tf$cast(n_classes, "float32") 
  print(class_loss %>% k_eval())
  
  list(1L, 1L)
}

# convert activations into bounding boxes
activations_to_bboxes <- function(activations, anchors) {
  
  # scale to between -1 and 1
  activations <- k_tanh(activations) 
  # move anchor boxes (centers as well as width/height) by scaled versions of the activations
  activation_centers <- (activations[ , 1:2]/2 * gridsize) + anchors[ , 1:2]
  activation_height_width <- (activations[ , 3:4]/2 + 1) * anchors[ , 3:4]
  
  # then convert to corners
  activation_corners <- hw2corners(activation_centers, activation_height_width)
  activation_corners
  
}



# Build model -------------------------------------------------------------

feature_extractor <- application_resnet50(include_top = FALSE, input_shape = c(224, 224, 3))

feature_extractor

input <- feature_extractor$input
common <- feature_extractor$output %>%
  layer_conv_2d(filters = 256, kernel_size = 3, padding = "same", name = "head_conv1") %>%
  layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, padding = "same", name = "head_conv2")

bbox_output <- layer_conv_2d(common, filters = 4, kernel_size = 3, padding = "same", name = "bbox_conv") %>%
  layer_reshape(target_shape = c(16, 4), name = "bbox_flatten")
class_output <- layer_conv_2d(common, filters = 1, kernel_size = 3, padding = "same", name = "class_conv") %>%
  layer_reshape(target_shape = c(16), name = "class_flatten")

model <-
  keras_model(inputs = input,
              outputs = list(bbox_output, class_output))

model %>% freeze_weights()
model
model %>% unfreeze_weights(from = "head_conv1")
model
       
model %>% compile(loss = list(bbox_loss, class_loss), optimizer = "adam")


