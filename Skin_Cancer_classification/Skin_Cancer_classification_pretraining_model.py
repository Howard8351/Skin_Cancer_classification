import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.data import Dataset
import numpy as np
import math
import pathlib
import os
import random
import time


#匯入類別資料
patient_data_path = "E:/GitHub Program/skin_cancer_data/HAM10000_metadata_oversample.csv"
#patient_data_path = "E:/GitHub Program/skin_cancer_data/HAM10000_metadata.csv"
patient_data  = pathlib.Path(patient_data_path).read_text()
#以分隔符號將資料分割並移除header
patient_data  = patient_data.split("\n")[1:]
#定義每個特徵的資料類別
col_data_type = [str(), str(), str(), str(), float(), str(), str()]
#根據給予的資料類別建立一個包含數個tensor的list
#每個tensor代表一個特徵
patient_data  = tf.io.decode_csv(patient_data, record_defaults = col_data_type)
    
number_of_class      = tf.unique(patient_data[2]).y.shape[0]
label                = tf.unique(patient_data[2]).idx
each_class_data_size = tf.unique_with_counts(patient_data[2]).count.numpy()
image_id             = patient_data[1]
#超參數設定
channels            = 3
epochs              = 50
batch_size          = 16
learning_rate       = 0.001
dataset_buffer_size = 10000
number_of_data      = label.shape[0]
input_image_size    = (200, 200)
data_path           = "E:/GitHub Program/skin_cancer_data/HAM10000_images_rename/"
    
class skin_cancer_classification_model:
    def __init__(self, image_id, label, number_of_class, each_class_data_size, channels, input_image_size,
                 epochs, batch_size, learning_rate, dataset_buffer_size, number_of_data, data_path):
        self.image_id              = image_id
        self.label                 = label
        self.number_of_class       = number_of_class
        self.channels              = channels
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.dataset_buffer_size   = dataset_buffer_size
        self.number_of_data        = number_of_data
        self.each_class_data_size  = each_class_data_size
        self.data_path             = data_path
        self.input_image_size      = input_image_size
        self.class_weight          = None
        self.model                 = None
        self.train_dataset         = None
        self.train_dataset_iterator= None
        self.train_data_size       = None
        self.train_step_each_epoch = None
        self.test_dataset          = None
        self.test_dataset_iterator = None
        self.test_data_size        = None
        self.test_step_each_epoch  = None
        self.test_data_list        = None
        self.model_test_label      = None
        self.model_train_label     = None
        #優化器設定
        self.optimizer             = tf.keras.optimizers.Adam(learning_rate)
        #損失函數設定
        self.loss_function         = tf.keras.losses.CategoricalCrossentropy()

    #建立訓練資料集名單
    def creat_data_list(self, training):
        #建立一個含整數的tensor方便合併選到的index
        train_data_index = tf.constant([0])
        test_data_index  = tf.constant([0])
        random.seed(10)
        #each_class_data_size = self.each_class_data_size / sum(self.each_class_data_size)
        #each_class_data_size = 1 - each_class_data_size
        class_weight_rate = [10., 1., 1., 10., 1., 5., 5.]
        class_weight = dict()

        for index in range(self.number_of_class):
            class_weight[index] = class_weight_rate[index]     
            class_index         = tf.math.equal(self.label, index)
            class_index         = tf.where(class_index)
            class_index         = tf.squeeze(class_index)
            class_index         = random.sample(class_index.numpy().tolist(), class_index.shape[0])
            train_index         = class_index[math.ceil(len(class_index) *0.2):len(class_index)]
            test_index          = class_index[0:math.ceil(len(class_index) *0.2)]
            train_data_index    = tf.concat([train_data_index, train_index], 0)
            test_data_index     = tf.concat([test_data_index, test_index], 0)

        self.class_weight = class_weight
        #把自己建立的數值移除
        split_tensor     = tf.split(train_data_index, [1, train_data_index.shape[0] - 1], 0)
        train_data_index = split_tensor[-1]
        split_tensor     = tf.split(test_data_index, [1, test_data_index.shape[0] - 1], 0)
        test_data_index  = split_tensor[-1]
        #取出選擇的資料
        train_image_id    = tf.convert_to_tensor(self.image_id.numpy()[train_data_index], tf.dtypes.string)
        train_image_label = tf.convert_to_tensor(self.label.numpy()[train_data_index])
        if training != True:
            self.model_train_label = train_image_label
        train_image_label = tf.strings.as_string(train_image_label)
        test_image_id     = tf.convert_to_tensor(self.image_id.numpy()[test_data_index], tf.dtypes.string)
        test_image_label  = tf.convert_to_tensor(self.label.numpy()[test_data_index])
        if training != True:
            self.model_test_label  = test_image_label
        test_image_label  = tf.strings.as_string(test_image_label)
        #建立選擇影像的路徑
        train_image_id  = tf.strings.join([train_image_id, train_image_label], "_")
        image_type      = ['.jpg'] * train_image_id.shape[0]
        train_image_id  = tf.strings.join([train_image_id, image_type])
        image_path      = [self.data_path] * train_image_id.shape[0]
        train_data_list = tf.strings.join([image_path, train_image_id])
        test_image_id   = tf.strings.join([test_image_id, test_image_label], "_")
        image_type      = ['.jpg'] * test_image_id.shape[0]
        test_image_id   = tf.strings.join([test_image_id, image_type])
        image_path      = [self.data_path] * test_image_id.shape[0]
        test_data_list = tf.strings.join([image_path, test_image_id])

        return train_data_list, test_data_list

    #dataset 處理函式
    def path_process(self, file_path):
        file_label = self.get_file_label(file_path)
        #匯入檔案
        image = tf.io.read_file(file_path)
        image = self.decode_image(image)
        return (image, file_label)

    #測試用dataset 處理函式(不回傳label)
    def model_test_path_process(self, file_path):
        #匯入檔案
        image = tf.io.read_file(file_path)
        image = self.decode_image(image)
        return image

    #取得檔案類別
    def get_file_label(self, file_path):
        #以python內建的檔案路徑分隔符號為標準分割字串
        split_string = tf.strings.split(file_path, "/")
        label        = tf.strings.split(split_string[-1], "_")
        label        = tf.strings.split(label[-1], ".")
        label        = tf.strings.to_number(label[0], out_type = tf.dtypes.int32)
        label        = tf.one_hot(label, self.number_of_class)
        #回傳檔案類別
        return label

    #將image傳成tensor
    def decode_image(self, image):
        #將image轉成nuit8 channel為3的tensor
        image = tf.io.decode_image(image, channels = 3)
        ##將image轉成指定的data type，並且作正規化
        #image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    #建立jpg影像dataset並回傳要幾次才能跑完一個epoch
    def creat_jpg_dataset(self, data_path_list, data_size, training = True, shuffle = True):
        dataset = Dataset.from_tensor_slices(data_path_list)
      
        if training:
            dataset = dataset.map(self.path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
            if shuffle:
                dataset = dataset.shuffle(data_size).repeat()
            else:
                dataset = dataset.repeat()
        else:
            dataset = dataset.map(self.model_test_path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        
        dataset         = dataset.batch(self.batch_size)
        step_each_epoch = math.ceil(data_size / self.batch_size)

        return dataset, step_each_epoch

    #建立分類模型
    def creat_model(self):
        
        input_layer   = Input(shape = self.input_image_size + (self.channels,))

        #使用ResNet101V2作為預先訓練網路
        #pretraining_model = tf.keras.applications.ResNet101V2(include_top = False,
        #                                                   input_shape = self.input_image_size + (self.channels,),
        #                                                   pooling = 'avg')
        #使用ResNet50V2作為預先訓練網路
        #pretraining_model = tf.keras.applications.ResNet50V2(include_top = False,
        #                                                     input_shape = self.input_image_size + (self.channels,),
        #                                                     pooling = 'avg')
        #pretraining_model = tf.keras.applications.InceptionResNetV2(include_top = False, input_tensor = input_layer)


        #使用EfficientNetB4作為預先訓練網路
        #輸入資料DataType須為Unit8
        pretraining_model = tf.keras.applications.EfficientNetB4(include_top = False, input_tensor = input_layer)
        #pretraining_model.trainable = False

        average_pool  = layers.GlobalAveragePooling2D(name = 'Top_layer_average_pool')(pretraining_model.output)
        bn_layer      = layers.BatchNormalization(name = 'Top_layer_batchnormalization')(average_pool)
        dropout_layer = layers.Dropout(0.2, name = 'Top_layer_dropout')(bn_layer)
        output_layer  = layers.Dense(self.number_of_class, activation = 'softmax', name = 'Top_layer_Dense')(dropout_layer)

        self.model    =  Model(input_layer, output_layer)
        self.model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics = "accuracy")

    #建立資料集
    def creat_dataset(self, training = True):
        train_data_list, test_data_list = self.creat_data_list(training)
        if training:
            shuffle = True
            #建立訓練資料集
            self.train_data_size = train_data_list.shape[0]
            self.train_dataset, self.train_step_each_epoch = self.creat_jpg_dataset(train_data_list, self.train_data_size, training, shuffle)
            #self.train_dataset_iterator = iter(self.train_dataset)
            #建立測試資料集
            shuffle = False
            self.test_data_size = test_data_list.shape[0]
            self.test_dataset, self.test_step_each_epoch = self.creat_jpg_dataset(test_data_list, self.test_data_size, training, shuffle)
            #self.test_dataset_iterator = iter(self.test_dataset)
        else:
            shuffle = False
            #建立訓練資料集
            self.train_data_size = train_data_list.shape[0]
            self.train_dataset, self.train_step_each_epoch = self.creat_jpg_dataset(train_data_list, self.train_data_size, training, shuffle)
            #建立測試資料集
            self.test_data_size = test_data_list.shape[0]
            self.test_dataset, self.test_step_each_epoch = self.creat_jpg_dataset(test_data_list, self.test_data_size, training, shuffle)

    #模型訓練主體
    def train_loop(self):
        self.creat_dataset(training = True)
        self.creat_model()
        self.model.summary()
        
        #建立callback
        callback_path              = "model_callback_output/model_weights"
        tensorboard_path           = "tensorboard_output" 
        model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath = callback_path, monitor = "val_loss", save_best_only = True,
                                                                        save_weights_only = True)
        tensorboard_callback       = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path, histogram_freq = 1)


        history = self.model.fit(x = self.train_dataset, steps_per_epoch = self.train_step_each_epoch, epochs = self.epochs,
                                  validation_data = self.test_dataset, validation_steps = self.test_step_each_epoch, 
                                  class_weight = self.class_weight, callbacks = [model_check_point_callback, tensorboard_callback])
    
    #模型再訓練訓練
    def load_model_and_training(self):
        self.creat_dataset(training = True)
        callback_path              = "model_callback_output"
        tensorboard_path           = "tensorboard_output" 
        self.creat_model()
        self.model.load_weights("model_callback_output")
        self.model.summary()
        #self.model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics = "accuracy")

        trainable_layer_name = ['Top_layer_average_pool', 'Top_layer_batchnormalization', 'Top_layer_dropout', 'Top_layer_Dense']
        #for layer_name in trainable_layer_name:
        #    layer = self.model.get_layer(layer_name)
        #    layer.trainable = True

        for index, layer in enumerate(self.model.layers):
            print(index, layer.name, layer.trainable)

        #建立callback
        model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath = callback_path, monitor = "val_loss", save_best_only = True,
                                                                        save_weights_only = True)
        tensorboard_callback       = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path, histogram_freq = 1)

        history = self.model.fit(x = self.train_dataset, steps_per_epoch = self.train_step_each_epoch, epochs = self.epochs,
                                  validation_data = self.test_dataset, validation_steps = self.test_step_each_epoch, 
                                  class_weight = self.class_weight,callbacks = [model_check_point_callback, tensorboard_callback])

    #載入模型並測試
    def load_model_and_test(self):
        callback_path = "model_callback_output/model_weights"
        self.creat_dataset(training = False)
        self.creat_model()
        self.model.load_weights(callback_path)
        
        train_data_label = tf.one_hot(self.model_train_label, self.number_of_class)
        train_data_number = tf.unique_with_counts(self.model_train_label).count.numpy()
        class_accuracy = [0.] * self.number_of_class

        model_predict = self.model.predict(x = self.train_dataset, steps = self.train_step_each_epoch)
        accuracy      = tf.keras.metrics.categorical_accuracy(train_data_label, model_predict)
        for index in range(self.model_train_label.shape[0]):
            class_accuracy[self.model_train_label[index].numpy()] += accuracy[index].numpy()
        class_accuracy = class_accuracy / train_data_number
        accuracy      = tf.math.reduce_mean(accuracy)
        print(f"Model train Accuracy:{accuracy}")
        for index in range(self.number_of_class):
            accuracy = class_accuracy[index]
            print(f"Class {index} train Accuracy: {accuracy}")

        class_accuracy = [0.] * self.number_of_class
        test_data_number  = tf.unique_with_counts(self.model_test_label).count.numpy()
        test_data_label = tf.one_hot(self.model_test_label, self.number_of_class)

        model_predict = self.model.predict(x = self.test_dataset, steps = self.test_step_each_epoch)
        accuracy      = tf.keras.metrics.categorical_accuracy(test_data_label, model_predict)
        for index in range(self.model_test_label.shape[0]):
            class_accuracy[self.model_test_label[index].numpy()] += accuracy[index].numpy()
        class_accuracy = class_accuracy / test_data_number
        accuracy      = tf.math.reduce_mean(accuracy)
        print(f"Model test Accuracy:{accuracy}")
        for index in range(self.number_of_class):
            accuracy = class_accuracy[index]
            print(f"Class {index} test Accuracy: {accuracy}")

skin_cancer_model = skin_cancer_classification_model(image_id, label, number_of_class, each_class_data_size,
                                                     channels, input_image_size, epochs, batch_size, learning_rate,
                                                     dataset_buffer_size, number_of_data, data_path)

##模型訓練
#skin_cancer_model.train_loop()
##模型再訓練
#skin_cancer_model.load_model_and_training()
#模型測試
skin_cancer_model.load_model_and_test()