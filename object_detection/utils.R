
load_and_preprocess_image <- function(image_name, target_height, target_width) {
  img_array <- image_load(file.path(img_dir, image_name), target_size = c(target_height, target_width)) %>%
    image_to_array() %>%
    imagenet_preprocess_input() 
  dim(img_array) <- c(1, dim(img_array))
  img_array
}


metric_iou <- function(y_true, y_pred) {
  # in this coordinate system, (0,0) is on the top left
  # the order is [x_left, y_top, x_right, y_bottom]
  intersection_xmin <- k_maximum(y_true[ ,1], y_pred[ ,1])
  intersection_ymin <- k_maximum(y_true[ ,2], y_pred[ ,2])
  intersection_xmax <- k_minimum(y_true[ ,3], y_pred[ ,3])
  intersection_ymax <- k_minimum(y_true[ ,4], y_pred[ ,4])
  area_intersection <- (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
  area_y <- (y_true[ ,3] - y_true[ ,1]) * (y_true[ ,4] - y_true[ ,2])
  area_yhat <- (y_pred[ ,3] - y_pred[ ,1]) * (y_pred[ ,4] - y_pred[ ,2])
  area_union <- area_y + area_yhat - area_intersection
  iou <- area_intersection/area_union
  k_mean(iou)
}


plot_image_with_boxes <- function(img_data,
                                  class_pred = NULL,
                                  box_pred = NULL) {
  img <- image_read(file.path(img_dir, img_data$file_name))
  print(img_data$name)
  img <- image_draw(img)
  rect(
    img_data$x_left,
    img_data$y_bottom,
    img_data$x_right,
    img_data$y_top,
    border = "cyan",
    lwd = 2.5
  )
  text(
    img_data$x_left,
    img_data$y_top,
    img_data$name,
    offset = 1,
    pos = 2,
    cex = 1.5,
    col = "cyan"
  )
  if (!is.null(box_pred))
    rect(preds[i, 1],
         preds[i, 2],
         preds[i, 3],
         preds[i, 4],
         border = "pink",
         lwd = 2.5)
  dev.off()
  print(img)
}

