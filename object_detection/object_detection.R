
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

imageinfo4ssd %>% arrange(desc(cnt)) # max is 37

max_pad <- 37

# Generator ---------------------------------------------------------------

ssd_generator <-
  function(data,
           max_pad,
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
      y_maxlen <- 4 * max_pad + max_pad # 185
      y <- array(0, dim = c(length(indices), y_maxlen))
      
      for (j in 1:length(indices)) {
        x[j, , ,] <-
          load_and_preprocess_image(data[[indices[j], "file_name"]], target_height, target_width)
        
        class_string <- data[indices[j], ]$categories
        classes <-  str_split(class_string, pattern = ", ")[[1]]
        n_classes <- length(classes)
        y[j, (y_maxlen - n_classes + 1):y_maxlen] <- classes
        
        xl_string <- data[indices[j], ]$xl
        yt_string <- data[indices[j], ]$yt
        xr_string <- data[indices[j], ]$xr
        yb_string <- data[indices[j], ]$yb
        
        xl <-  str_split(xl_string, pattern = ", ")[[1]] %>% as.integer()
        yt <-  str_split(yt_string, pattern = ", ")[[1]] %>% as.integer()
        xr <-  str_split(xr_string, pattern = ", ")[[1]] %>% as.integer()
        yb <-  str_split(yb_string, pattern = ", ")[[1]] %>% as.integer()
        
        boxes_maxlen <- 4 * max_pad
        boxes <- c(rbind(xl,yt,xr, yb)) 
        y[j, (boxes_maxlen - 4 * n_classes + 1):boxes_maxlen] <- boxes
         
      }
      x <- x/255
      list(x, y)
    }
  }

train_gen <- ssd_generator(
  imageinfo4ssd,
  max_pad = 37,
  target_height = target_height,
  target_width = target_width,
  shuffle = TRUE,
  batch_size = batch_size
)

batch <- train_gen()
c(x,y) %<-% train_gen()
x %>% dim()
y

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

# from here on, everything needs to be callable from the loss function
anchor_centers <- cbind(anchor_xs, anchor_ys) %>% k_constant() 
anchor_height_width <- matrix(1/gridsize, nrow = 16, ncol = 2) %>% k_constant()
# anchor information is x/y coords of center, height, width
anchors <- k_concatenate(list(anchor_centers, anchor_height_width))
ggplot(data.frame(x = anchor_xs, y = anchor_ys), aes(x, y)) + geom_point() + theme(aspect.ratio = 1)

# convert from center-height-width representation to grid corners
hw2corners <- function(centers, height_width) {
  k_concatenate(list(centers - height_width/2, centers + height_width/2))
}
anchor_corners <- hw2corners(anchor_centers, anchor_height_width)


# SSD loss ----------------------------------------------------------------

# image size
size <- 224

# predictions will come in with shape (batchsize, 16, 25)
# 16 is number of grid cells
# 25 is concatenation of 21 and 4
# 21 class preds per box
# 4 bbox coords per box

# ground truth will come in with shape (batchsize, n * (1 + 4))
# where n is maximum number of detections in image


### overall loss
ssd_loss <- function(y_true, y_pred) {
  
  loc_loss_batch <- k_constant(0)
  class_loss_batch <- k_constant(0)
  
  # loop over batch
  for (i in 1:nrow(y_true)) {
    
    # deconstruct ground truth and predictions into bounding boxes and class predictions
    y_true_bbox <- y_true[i, 1:12]
    y_true_class <- y_true[i, 13:15]

    y_pred_bbox <- y_pred[i, , 1:4] 
    y_pred_class <- y_pred[i, , 5:25]
    
    losses <- ssd_loss_single(y_true_bbox, y_true_class, y_pred_bbox, y_pred_class)
    loc_loss_batch <- loc_loss_batch + losses[[1]]
    class_loss_batch <- class_loss_batch + losses[[2]]
  }
  
  loc_loss_batch + class_loss_batch

}

### loss per image
ssd_loss_single <- function(y_true_bbox, y_true_class, y_pred_bbox, y_pred_class) {
  
  # convert activations into bounding boxes
  acts <- activations_to_bboxes(y_pred_bbox, anchors)
  
  # remove padding from ground truth
  y_extracted <- get_y(y_true_bbox, y_true_class)
  y_true_bbox <- y_extracted[ , 1:4]
  y_true_class <- y_extracted[ , 5]
  
  # compute IOU between grid cells and ground truth boxes
  overlaps <- jaccard(y_true_bbox, anchor_corners)
  #print(overlaps %>% k_eval())
  
  # "matching problem":
  # determine best overlap of ground truth boxes and anchor (prior) boxes (grid cells, here)
  overlaps <- map_to_ground_truth(overlaps)
  #print(overlaps)
  gt_overlap <- overlaps[1:16]
  gt_idx <- overlaps[17:32] %>% k_cast("int32")
  #print(gt_overlap %>% k_eval())
  
  #print(y_true_class %>% k_eval())
  #print(gt_idx %>% k_eval())
  # find matching class from ground truth class vector
  # each grid cell gets assigned a class here
  gt_class <- k_gather(y_true_class, gt_idx) %>% tf$cast("int32")
  #print(gt_class %>% k_eval())
  class_len <- length(gt_class)
  #print(class_len)
  #print(n_classes)
  
  # pull out grid cells with overlap > 0
  pos_idx <- tf$where(tf$greater(gt_overlap, 0.4))[ , 1] # 0-based
  #print(pos_idx %>% k_eval())
  
  # set all cells with low overlap to background class
  gt_class <- tf$where(tf$less(gt_overlap, 0.4) , tf$fill(list(class_len), tf$constant(class_background)), gt_class)
  #print(gt_class %>% k_eval())
  
  #print(y_true_bbox %>% k_eval())
  #print(gt_idx %>% k_eval()) ####### indexing problem!!
  # pick ground truth bounding boxes for detected objects
  gt_bbox <- k_gather(y_true_bbox, gt_idx)
  #print(gt_bbox %>% k_eval())  
  
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

# remove padding from ground truth
get_y <- function(true_bbox,true_class) {
  
  true_bbox <- k_reshape(true_bbox, c(-1, 4))
  #print(true_bbox %>% k_eval())
  # normalize bounding boxes to between 0 and 1
  true_bbox <- true_bbox/size
  # only keep nonzero entries
  mask <- tf$greater(true_bbox[ , 3] - true_bbox[ , 1], 0) # height-xcenter > 0
  true_bbox_keep <- tf$boolean_mask(true_bbox, mask)
  true_class_keep <- tf$boolean_mask(true_class, mask) %>% k_reshape(c(-1, 1))
  #print(true_bbox_keep %>% k_eval())
  #print(true_class_keep %>% k_eval())
  concat <- k_concatenate(list(true_bbox_keep, true_class_keep))
  # print(concat %>% k_eval())
  concat
}


# compute IOU
jaccard <- function(bbox, anchor_corners) {
  intersection <- intersect(bbox, anchor_corners)
  #print(intersection %>% k_eval())
  union <- k_expand_dims(box_area(bbox), axis = 2)  + k_expand_dims(box_area(anchor_corners), axis = 1) - intersection
  #print(union %>% k_eval())
  intersection/union
}

# compute intersection for IOU
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

area <- function(box) {
  (box[ , 3] - box[ , 1]) * (box[ , 4] - box[ , 2]) ## these are conrners!
} 

# determine best overlap of ground truth boxes and anchor (prior) boxes 
map_to_ground_truth <- function(overlaps) {
  
  # overlaps shape is: number of ground truth objects * number of grid cells
  #print(overlaps %>% k_eval())
  # for each ground truth object, find maximally overlapping cell (crit. 1)
  # shape: number of ground truth objects
  prior_overlap <- k_max(overlaps, axis = 2)
  #print(prior_overlap %>% k_eval())
  prior_idx <- k_argmax(overlaps, axis = 2) 
  #print(prior_idx %>% k_eval())
  
  # for each grid cell, what object does it overlap with most (crit. 2)
  # shape: number of grid cells
  gt_overlap <- k_max(overlaps, axis = 1)
  gt_idx <- k_argmax(overlaps, axis = 1)
  #print(gt_idx %>% k_eval())
  
  # boolean representation of maximally overlapping anchor
  prior_idx_bool <- prior_idx %>%
    tf$one_hot(depth = k_shape(overlaps)[2]) %>%
    k_sum(axis = 1) %>%
    k_cast(tf$bool)
  
  #print(prior_idx_bool %>% k_eval())
  
  replacement <- tf$fill(list(k_shape(overlaps)[2]), 1.99)
  #print(replacement %>% k_eval())
  
  #print(gt_overlap %>% k_eval())  
  # set all definitely overlapping cells to respective object (crit. 1)
  gt_overlap <- tf$where(prior_idx_bool, replacement, gt_overlap)
  #print(gt_overlap %>% k_eval())

  # now still set all others to best match by crit. 2
  # tbd
  #
  #for(i in 1:length(prior_idx_eval)) {
  #  p <- prior_idx_eval[i]
  #  gt_idx_eval[p] <- i
  #}
  
  gt_idx <- k_constant(as.integer(c(2, 1, 1, 2, 1, 1, 3, 2, 1, 0, 0, 2, 0, 0, 0, 2)))
  gt_idx <- gt_idx %>% k_cast("float32")
  # return: vectors of length (number of grid cells), containing measures of overlap as well as object indices
  k_concatenate(list(gt_overlap, gt_idx))

}



n_classes <- tf$constant(21L, dtype = "int32")

ssd_loss(test_y_true, test_y_pred) %>% k_eval()



test_y_pred <- array(runif(batch_size * 16 * 25), dim = c(batch_size, 16, 25)) %>% k_constant()

test_y_true_cl <- matrix(c(1,14,14,0,11,17), nrow = 2, ncol = 3, byrow = TRUE)
test_y_true_bb <- matrix(c(91,47,223,169,0,49,205,180,9,169,217,222,
                           0,0,0,0,81,49,166,171,63,1,222,223), nrow = 2, ncol = 12, byrow = TRUE)
test_y_true <- cbind(test_y_true_bb, test_y_true_cl) %>% k_constant()



# Build model -------------------------------------------------------------



n_anchors_per_cell <- 1

n_filters <- n_anchors_per_cell * (4 + n_classes)


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


model %>% freeze_weights()
model
model %>% unfreeze_weights(from = "head_conv1")
model
       


############ tbd with model predictions



##### ground truth
# idx is index into batch
# bbox,clas = get_y(y[0][idx], y[1][idx]) # ground truth
# bbox,clas sizes: 3*4, 3

##### model output

## before

# batch = learn.model(x)
# b_clas,b_bb = batch
# b_clasi = b_clas[idx] 
# b_bboxi = b_bb[idx]

# Here is our 4x4 grid cells from our final convolutional layer 
#torch_gt(ax, ima, anchor_cnr, b_clasi.max(1)[1])

## now: matching problem

# we are going to go through a matching problem where we are going to take every one of these 16 boxes and see which one of these three ground truth objects has the highest amount of overlap with a given square

# a_ic = actn_to_bb(b_bboxi, anchors) 
# overlaps = jaccard(bbox.data, anchor_cnr.data) # 3x16

# for each ground truth object, find maximally overlapping cell
# overlaps.max(1)
# for each grid cell, what object does it overlap with most 
# overlaps.max(0)

# The way it assign that is each of the three (row-wise max) gets assigned as is. For the rest of the anchor boxes, they get assigned to anything which they have an overlap of at least 0.5 with (column-wise). If neither applies, it is considered to be a cell which contains background.
# gt_overlap,gt_idx = map_to_ground_truth(overlaps)

#  Now we can combine these values to classes:
# gt_clas = clas[gt_idx]; gt_clas

# Then add a threshold and finally comes up with the three classes that are being predicted:
#thresh = 0.5
#pos = gt_overlap > thresh
#pos_idx = torch.nonzero(pos)[:,0]
#neg_idx = torch.nonzero(1-pos)[:,0]
#pos_idx
#gt_clas[1-pos] = len(id2cat)
#[id2cat[o] if o<len(id2cat) else 'bg' for o in gt_clas.data]