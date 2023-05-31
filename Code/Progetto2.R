
original_dataset_dir_tr_O <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TRAIN/O"   
original_dataset_dir_tr_R <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TRAIN/R"  
original_dataset_dir_te_O <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TEST/O"   
original_dataset_dir_te_R <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TEST/R"  
base_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2"  
dir.create(base_dir)

#Creazione delle cartelle nella nuova cartella appena creata
train_dir <- file.path(base_dir, "train") 
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_O_dir <- file.path(train_dir, "O")
dir.create(train_O_dir)
train_R_dir <- file.path(train_dir, "R")
dir.create(train_R_dir)
validation_O_dir <- file.path(validation_dir, "O")
dir.create(validation_O_dir)
validation_R_dir <- file.path(validation_dir, "R")
dir.create(validation_R_dir)
test_O_dir <- file.path(test_dir, "O")
dir.create(test_O_dir)
test_R_dir <- file.path(test_dir, "R")
dir.create(test_R_dir)


set.seed(123)
ind_tr <- matrix(c(1:13968))
tr_ind_O <- sample(c(1:201,203:4429,4431:13968), 8000)
val_ind_O <- sample(ind_tr[-c(tr_ind_O,202,4430),],2000)
te_ind_O <- sample(ind_tr[-c(val_ind_O,tr_ind_O,202,4430),],1000)
tr_ind_R <- sample(1:11111, 8000)
val_ind_R <- sample(matrix(1:11111)[-tr_ind_R,], 2000)
te_ind_R <- sample(matrix(1:11111)[-c(tr_ind_R,val_ind_R),],1000)



fnames <- paste0("O_", tr_ind_O, ".jpg")
for (i in tr_ind_O) {
  if(i<=12567) file.copy(file.path(original_dataset_dir_tr_O, fnames[which(tr_ind_O==i)]),file.path(train_O_dir))
  else file.copy(file.path(original_dataset_dir_te_O, fnames[which(tr_ind_O==i)]),file.path(train_O_dir))
}
fnames <- paste0("O_", val_ind_O, ".jpg")
for (i in val_ind_O) {
  if(i<=12567) file.copy(file.path(original_dataset_dir_tr_O, fnames[which(val_ind_O==i)]),file.path(validation_O_dir))
  else file.copy(file.path(original_dataset_dir_te_O, fnames[which(val_ind_O==i)]),file.path(validation_O_dir))
}
fnames <- paste0("O_", te_ind_O, ".jpg")
for (i in te_ind_O) {
  if(i<=12567) file.copy(file.path(original_dataset_dir_tr_O, fnames[which(te_ind_O==i)]),file.path(test_O_dir))
  else file.copy(file.path(original_dataset_dir_te_O, fnames[which(te_ind_O==i)]),file.path(test_O_dir))
}
fnames <- paste0("R_", tr_ind_R, ".jpg")
for (i in tr_ind_R) {
  if(i<=9999) file.copy(file.path(original_dataset_dir_tr_R, fnames[which(tr_ind_R==i)]),file.path(train_R_dir))
  else file.copy(file.path(original_dataset_dir_te_R, fnames[which(tr_ind_R==i)]),file.path(train_R_dir))
}
fnames <- paste0("R_", val_ind_R, ".jpg")
for (i in val_ind_R) {
  if(i<=9999) file.copy(file.path(original_dataset_dir_tr_R, fnames[which(val_ind_R==i)]),file.path(validation_R_dir))
  else file.copy(file.path(original_dataset_dir_te_R, fnames[which(val_ind_R==i)]),file.path(validation_R_dir))
}
fnames <- paste0("R_", te_ind_R, ".jpg")
for (i in te_ind_R) {
  if(i<=9999) file.copy(file.path(original_dataset_dir_tr_R, fnames[which(te_ind_R==i)]),file.path(test_R_dir))
  else file.copy(file.path(original_dataset_dir_te_R, fnames[which(te_ind_R==i)]),file.path(test_R_dir))
}


cat("total training O images:", length(list.files(train_O_dir)), "\n")
cat("total training R images:", length(list.files(train_R_dir)), "\n")
cat("total validation O images:", length(list.files(validation_O_dir)), "\n")
cat("total validation R images:", length(list.files(validation_R_dir)), "\n")
cat("total test O images:", length(list.files(test_O_dir)), "\n")
cat("total test R images:", length(list.files(test_R_dir)), "\n")



################################################################################################
################################################################################################
#         FFN
################################################################################################
################################################################################################

early_stopping <- callback_early_stopping(monitor = "val_accuracy",
                                          min_delta = 0.005,
                                          patience = 5, mode = "max")

library(keras)

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

train_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/train"

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(224,224),
  batch_size = 128,
  class_mode = "binary"
)


validation_dir  <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/validation"

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(224,224),
  batch_size = 32,
  class_mode = "binary"
)  


#FFN
model_FFN <- keras_model_sequential() %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


model_FFN %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


early_stopping <- callback_early_stopping(monitor = "val_accuracy",
                                          min_delta = 0.005,
                                          patience = 5, mode = "max")

history_FFN <- model_FFN %>% fit(
  train_generator, callbacks = early_stopping,
  epochs = 100,
  steps_per_epoch = 250,
  validation_data = validation_generator,
  validation_steps = 125
)

model_FFN %>% save_model_hdf5("Waste_Classification_FFN.h5")

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(224, 224),
  batch_size = 20,
  class_mode = "binary"
)
model_FFN %>% evaluate_generator(test_generator, steps = 50) #acc 76%





################################################################################################
#########################################################################################
#          CNN            #########################################################################################
#########################################################################################
#########################################################################################


train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)
train_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/train"

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen, 
  target_size = c(224, 224),
  batch_size = 128,
  class_mode = "binary"
)

validation_dir  <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/validation"

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(224,224),
  batch_size = 32,
  class_mode = "binary"
)  

#CNN
model_CNN <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(224, 224, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


model_CNN %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)

early_stopping <- callback_early_stopping(monitor = "val_accuracy",
                                          min_delta = 0.005,
                                          patience = 5, mode = "max")

history_CNN <- model_CNN %>% fit_generator(
  train_generator, callbacks = early_stopping,
  steps_per_epoch = 125,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 125
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(224, 224),
  batch_size = 20,
  class_mode = "binary"
)

plot_CNN <- plot(history_CNN)
model_CNN %>% evaluate_generator(test_generator, steps = 100) #84%





################################################################################################
#########################################################################################
#          DATA AUGMENTATION            #########################################################################################
#########################################################################################
#########################################################################################

#DATA AUGMENTATION
#train_O_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/train/O"
#fnames <- list.files(train_O_dir, full.names = TRUE)
#img_path <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati2/train/O/O_407.jpg"
#img <- image_load(img_path, target_size=c(28,28))
#img_array <- image_to_array(img)
#img_array <- array_reshape(img_array, c(1, 28, 28, 3))
#data_generator <- flow_images_from_data(img_array, generator = train_datagen, batch_size = 1)
#batch <- generator_next(data_generator)
#plot(as.raster(batch[1,,,]))

tic()
data_aug <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  fill_mode = "nearest"
)

train_generator <- flow_images_from_directory(
  train_dir,
  data_aug,
  target_size = c(224, 224),
  batch_size = 128,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(224, 224),
  batch_size = 32,
  class_mode = "binary"
)

model_AUG <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(224, 224, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


model_AUG %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)

early_stopping <- callback_early_stopping(monitor = "val_accuracy",
                                          min_delta = 0.005,
                                          patience = 5, mode = "max")

history_AUG <- model_AUG %>% fit_generator(
  train_generator, callbacks = early_stopping,
  steps_per_epoch = 125,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 125
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(224, 224),
  batch_size = 20,
  class_mode = "binary"
)

plot_AUG <- plot(history_AUG)
model_AUG %>% evaluate_generator(test_generator, steps = 100) #86.6%



################################################################################################
#########################################################################################
#          VGG16            #########################################################################################
#########################################################################################
#########################################################################################
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
#imagenet_preprocess_input()
datagen <- image_data_generator(rescale = 1/255)
extract_features <- function(directory, sample_count, batch_size) {
  
  features <- array(0, dim = c(sample_count, 7, 7, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(224, 224),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    
    if (i * batch_size >= sample_count)
      break 
  }
  
  list(
    features = features,
    labels = labels
  ) 
}


train <- extract_features(train_dir, 16000, 64)
validation <- extract_features(validation_dir, 4000, 32)
test <- extract_features(test_dir, 2000, 20)


reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 7 * 7 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model_VGG16 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu",
              input_shape = 7 * 7 * 512) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_VGG16 %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


early_stopping <- callback_early_stopping(monitor = "val_accuracy",
                                          min_delta = 0.005,
                                          patience = 5, mode = "max", 
                                          restore_best_weights = TRUE)

history_VGG16 <- model_VGG16 %>% fit(
  train$features, train$labels,
  callbacks = early_stopping,
  epochs = 30,
  batch_size = 32,
  validation_data = list(validation$features, validation$labels)
)

model_VGG16 %>% save_model_hdf5("Waste_Classification_VGG16.h5")
plot_VGG16 <- plot(history_VGG16) 

model_VGG16 %>% evaluate(test$features, test$labels)

pred <- model_VGG16 %>% predict(test$features)
pred <- ifelse(pred <=0.5,0,1)
library(caret)
confusionMatrix(data = factor(as.vector(pred)), reference = factor(test$labels))
?confusionMatrix

