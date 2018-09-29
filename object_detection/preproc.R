library(keras)
library(rjson)
library(magick)
library(purrr)
library(tibble)
library(tidyr)
library(dplyr)

img_dir <- "data/VOCdevkit/VOC2007/JPEGImages"
annot_file <- "data/pascal_train2007.json"

annotations <- fromJSON(file = annot_file)
str(annotations, max.level = 1)

imageinfo <- annotations$images %>% {
  tibble(
    id = map_dbl(., "id"),
    file_name = map_chr(., "file_name"),
    image_height = map_dbl(., "height"),
    image_width = map_dbl(., "width"),
    
    )
  }

boxinfo <- annotations$annotations %>% {
  tibble(
    image_id = map_dbl(., "image_id"),
    category_id = map_dbl(., "category_id"),
    bbox = map(., "bbox")
    
  )
}

# note: y starts from top!
boxinfo <- boxinfo %>% mutate(bbox = unlist(map(.$bbox, function(x) paste(x, collapse = " "))))
boxinfo <- boxinfo %>% separate(bbox, into = c("x_left", "y_top", "bbox_width", "bbox_height"))
boxinfo <- boxinfo %>% mutate_all(as.numeric)
boxinfo <- boxinfo %>% mutate(y_bottom = y_top + bbox_height - 1, x_right = x_left + bbox_width - 1)
boxinfo <- boxinfo %>% mutate(area = bbox_width * bbox_height)

catinfo <- annotations$categories %>%  {
  tibble(
  id = map_dbl(., "id"),
  name = map_chr(., "name")
  )
}

imageinfo <- imageinfo %>% 
  inner_join(boxinfo, by = c("id" = "image_id")) %>%
  inner_join(catinfo, by = c("category_id" = "id"))

nrow(imageinfo)

img_data <- imageinfo[1, ]
img <- image_read(file.path(img_dir, img_data$file_name))
img <- image_draw(img)
rect(img_data$x_left, img_data$y_bottom,img_data$x_right, img_data$y_top, border = "white", lwd = 2)
text(img_data$x_left, img_data$y_top, img_data$name, offset = 1, pos = 2, cex = 1.5, col = "white")
dev.off()
print(img)


