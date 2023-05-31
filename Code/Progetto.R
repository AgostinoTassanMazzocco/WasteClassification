#PROGETTO

original_dataset_dir_tr_O <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TRAIN/O"  #Directory dataset originale 
original_dataset_dir_tr_R <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TRAIN/R"  #Directory dataset originale
original_dataset_dir_te_O <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TEST/O"   #Directory dataset originale
original_dataset_dir_te_R <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/OriginalDirectory/TEST/R"   #Directory dataset originale
base_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati"  #Directory cartella in cui salvare tutti i nuovi dati (non deve esistere già, il comando successivo crea la cartella)
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

#Campionamento dalle immagini del training delle immagini da inserire nel validation set 
set.seed(123)
v_ind_O <- sample(1:12567, 1401)
tr_ind_O <- matrix(1:12567)[-v_ind_O,]
v_ind_R <- sample(1:9999,1112)
tr_ind_R <- matrix(1:9999)[-v_ind_R,]

#Copia Incolla delle immagini dal dataset originale alle nuove cartelle 
fnames <- paste0("O_", tr_ind_O, ".jpg")
file.copy(file.path(original_dataset_dir_tr_O, fnames),
          file.path(train_O_dir)) #Qua dà errore perchè le immagini O_202 e un'altra non esistono ma non è importante 
fnames <- paste0("R_", tr_ind_R, ".jpg")
file.copy(file.path(original_dataset_dir_tr_R, fnames),
          file.path(train_R_dir))

fnames <- paste0("O_", v_ind_O, ".jpg")
file.copy(file.path(original_dataset_dir_tr_O, fnames),
          file.path(validation_O_dir))
fnames <- paste0("R_", v_ind_R, ".jpg")
file.copy(file.path(original_dataset_dir_tr_R, fnames),
          file.path(validation_R_dir))

fnames <- paste0("O_", 12568:13968, ".jpg")
file.copy(file.path(original_dataset_dir_te_O, fnames),
          file.path(test_O_dir))
fnames <- paste0("R_", 10000:11111, ".jpg")
file.copy(file.path(original_dataset_dir_te_R, fnames),
          file.path(test_R_dir))

#Comandi che stampano a video il numero di immagini di ognuna delle cartelle create
cat("total training O images:", length(list.files(train_O_dir)), "\n")
cat("total training R images:", length(list.files(train_R_dir)), "\n")
cat("total validation O images:", length(list.files(validation_O_dir)), "\n")
cat("total validation R images:", length(list.files(validation_R_dir)), "\n")
cat("total test O images:", length(list.files(test_O_dir)), "\n")
cat("total test R images:", length(list.files(test_R_dir)), "\n")


#Preparazione dei dati
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)
train_dir <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati/train"

train_generator <- flow_images_from_directory(
  # target directory
  train_dir,
  # target data generator
  train_datagen,
  # Tutte le immagini sono riscalate tra  
  target_size = c(100, 100),
  batch_size = 20,
  # Utilizziamo questo tipo di classe perché necessitiamo di etichette binarie
  class_mode = "binary"
)

validation_dir  <- "/Users/Ago/Desktop/Magistrale/Data Mining/Intro to Deep Learning/Progetto/Dati/validation"

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(100,100),
  batch_size = 20,
  class_mode = "binary"
)  
#Softmax -> 2 units and categorical_crossentropy
#Sigmoid -> 1 unit and binary_crossentropy

#FFN 
model <- keras_model_sequential() %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(100*100,3)) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = optimizer_rmsprop(lr = 1e-4),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  # Numero di campioni da cosiderare per ogni epoca
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  # Definisce quanti batches prendere dal validation generator
  validation_steps = 50
)


test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(100, 100),
  batch_size = 20,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)
#Accuracy 86%









#CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

history <- model %>% fit_generator(
  train_generator,
  # Numero di campioni da cosiderare per ogni epoca
  steps_per_epoch = 50,
  epochs = 10,
  validation_data = validation_generator,
  # Definisce quanti batches prendere dal validation generator
  validation_steps = 20
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)




