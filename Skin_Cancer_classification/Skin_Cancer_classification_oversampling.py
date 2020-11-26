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
patient_data_path = "E:/GitHub Program/skin_cancer_data/HAM10000_metadata.csv"
patient_data  = pathlib.Path(patient_data_path).read_text()
#以分隔符號將資料分割並移除header
patient_data  = patient_data.split("\n")[1:]
#定義每個特徵的資料類別
col_data_type = [str(), str(), str(), str(), float(), str(), str()]
#根據給予的資料類別建立一個包含數個tensor的list
#每個tensor代表一個特徵
patient_data  = tf.io.decode_csv(patient_data, record_defaults = col_data_type)
    
number_of_class      = tf.unique(patient_data[2]).y.shape[0]
#每種class的資料都被oversampling到6705筆
each_class_data_size = [6705] * number_of_class
#超參數設定
channels            = 3
epochs              = 20
batch_size          = 16
learning_rate       = 0.0001
dataset_buffer_size = 1500
data_size           = 46935
data_path           = "E:/GitHub Program/skin_cancer_data/HAM10000_images_rename/"
    
class skin_cancer_classification_model:
    def __init__(self, number_of_class, each_class_data_size, channels, 
                 epochs, batch_size, learning_rate, dataset_buffer_size, data_size, data_path):
        #self.image_id              = image_id
        #self.label                 = label
        self.number_of_class       = number_of_class
        self.channels              = channels
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.dataset_buffer_size   = dataset_buffer_size
        #self.data_size             = data_size
        self.each_class_data_size  = each_class_data_size
        self.train_data_size       = None
        self.data_path             = data_path
        self.start_nodes           = 64
        self.model                 = None
        self.train_dataset         = None
        self.train_dataset_iterator= None
        self.train_data_size       = None
        self.train_step_each_epoch = None
        self.test_dataset          = None
        self.test_dataset_iterator = None
        self.test_data_size        = None
        self.test_step_each_epoch  = None
        self.loss_value            = None
        self.best_accuracy         = None
        #優化器設定
        self.optimizer             = tf.keras.optimizers.Adam(learning_rate)
        #損失函數設定
        self.loss_function         = tf.keras.losses.CategoricalCrossentropy()

    #建立訓練資料集名單
    def creat_oversampling_data_list(self):
        #以list匯入檔名
        image_id = os.listdir(self.data_path)
        random.seed(10)
        #將影像名稱隨機排序
        random_sort_image_id = random.sample(image_id, len(image_id))
        #前八成做訓練資料集
        train_data_list = random_sort_image_id[0:math.ceil(len(image_id) * 0.8)]
        train_data_list = tf.convert_to_tensor(train_data_list)
        #後兩成做訓練資料集
        test_data_list = random_sort_image_id[math.ceil(len(image_id) * 0.8) : len(image_id)]
        test_data_list = tf.convert_to_tensor(test_data_list)

        #建立選擇影像的路徑
        image_path = [self.data_path] * train_data_list.shape[0]
        train_data_list = tf.strings.join([image_path, train_data_list])
        image_path = [self.data_path] * test_data_list.shape[0]
        test_data_list = tf.strings.join([image_path, test_data_list])

        return train_data_list, test_data_list

    #dataset 處理函式
    def path_process(self, file_path):
        file_label = self.get_file_label(file_path)
        #匯入檔案
        image = tf.io.read_file(file_path)
        image = self.decode_image(image)
        return file_label, image

    #取得檔案類別
    def get_file_label(self, file_path):
        #以python內建的檔案路徑分隔符號為標準分割字串
        split_string = tf.strings.split(file_path, "/")
        label        = tf.strings.split(split_string[-1], "_")
        label        = tf.strings.split(label[-1], ".")
        #回傳檔案類別
        return label[0]

    #將image傳成tensor
    def decode_image(self, image):
        #將image轉成nuit8 channel為3的tensor
        image = tf.io.decode_image(image, channels = 3)
        #將image轉成指定的data type，並且作正規化
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    #建立jpg影像dataset並回傳要幾次才能跑完一個epoch
    def creat_jpg_dataset(self, data_path_list, data_size):
        data_list = Dataset.list_files(data_path_list, shuffle = False)
        dataset   = data_list.map(self.path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset   = dataset.shuffle(self.dataset_buffer_size).repeat()
        dataset   = dataset.batch(self.batch_size)
        step_each_epoch = math.ceil(data_size / self.batch_size)

        return dataset, step_each_epoch

    #建立分類模型
    def creat_model(self):
        #測試使用類似ResNet架構
            
        input_layer   = Input(shape = [400, 400, self.channels, ], name = "Input_layer")
        #第一區多用較大的stride降維
        block_0_con2d_layer   = layers.Conv2D(self.start_nodes, 3, 2, name = "block_0_con2d_layer")(input_layer)
        block_0_bn_layer      = layers.BatchNormalization(name = "block_0_bn_layer")(block_0_con2d_layer)
        block_0_act_layer     = layers.PReLU(name = "block_0_act_layer")(block_0_bn_layer)
        block_0_maxpool_layer = layers.MaxPool2D(name = "block_0_maxpool_layer")(block_0_act_layer)
        #第二區由數個block組成
        #通常會在每個block開頭先降解析度並增加維度
        block_1_reduce_image_size = self.reduce_image_size(block_0_maxpool_layer, 1)
        block_1_main_con2d_net    = self.con2d_net_block(block_1_reduce_image_size, 1, 2)

        block_2_reduce_image_size = self.reduce_image_size(block_1_main_con2d_net, 2)
        block_2_main_con2d_net    = self.con2d_net_block(block_2_reduce_image_size, 2, 2)

        block_3_reduce_image_size = self.reduce_image_size(block_2_main_con2d_net, 3)
        block_3_main_con2d_net    = self.con2d_net_block(block_3_reduce_image_size, 3, 3)

        block_4_reduce_image_size = self.reduce_image_size(block_3_main_con2d_net, 4)
        block_4_main_con2d_net    = self.con2d_net_block(block_4_reduce_image_size, 4, 5)

        block_5_reduce_image_size = self.reduce_image_size(block_4_main_con2d_net, 5)
        block_5_main_con2d_net    = self.con2d_net_block(block_5_reduce_image_size, 5, 2)

        #最後以一個avg_pool降維並連接到輸出
        avg_pool_layer = layers.AvgPool2D(4, 1)(block_5_main_con2d_net)
        flatten_layer  = layers.Flatten()(avg_pool_layer)
        output_layer   = layers.Dense(self.number_of_class, activation = 'softmax')(flatten_layer)

        self.model    =  Model(input_layer, output_layer)

    #用卷機網路降低影像解析度
    def reduce_image_size(self, input_layer, block = 1):
        #未來考慮加入名稱增強架構辨識度
        block_name = "block_" + str(block)
        reduce_size_con2d_layer    = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes), 1, 2,
                                                       name = block_name + "_reduce_size_con2d_layer")(input_layer)
        reduce_size_bn_layer       = layers.BatchNormalization()(reduce_size_con2d_layer)
        reduce_size_act_layer      = layers.PReLU()(reduce_size_bn_layer)
            
        con2d_layer = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes), 3, 1, "same")(reduce_size_act_layer)
        bn_layer    = layers.BatchNormalization()(con2d_layer)
        act_layer   = layers.PReLU()(bn_layer)
            
        increase_channels_con2d    = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes) * 4, 1, 1)(act_layer)
        increase_channels_bn_layer = layers.BatchNormalization()(increase_channels_con2d)

        residual_con2d_layer = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes) * 4, 1, 2)(input_layer)
        residual_bn_layer    = layers.BatchNormalization()(residual_con2d_layer)

        add_layer        = layers.add([increase_channels_bn_layer, residual_bn_layer])
        output_act_layer = layers.PReLU()(add_layer)

        return output_act_layer

    #卷機網路主要區塊
    def con2d_net_block(self, input_layer, block = 1, repeat_times = 1):
        for i in range(repeat_times):
            con2d_layer = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes), 1, 1)(input_layer)
            bn_layer    = layers.BatchNormalization()(con2d_layer)
            act_layer   = layers.PReLU()(bn_layer)

            con2d_layer = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes), 3, 1, 'same')(act_layer)
            bn_layer    = layers.BatchNormalization()(con2d_layer)
            act_layer   = layers.PReLU()(bn_layer)

            increase_channels_con2d    = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes) * 4, 1, 1)(act_layer)
            increase_channels_bn_layer = layers.BatchNormalization()(increase_channels_con2d)

            residual_con2d_layer = layers.Conv2D(int(math.pow(2, block - 1) * self.start_nodes) * 4, 1, 1)(input_layer)
            residual_bn_layer    = layers.BatchNormalization()(residual_con2d_layer)

            add_layer   = layers.add([increase_channels_bn_layer, residual_bn_layer])
            #為方便做迴圈
            input_layer = layers.PReLU()(add_layer)

        return input_layer

    #建立資料集
    def creat_dataset(self):
        train_data_list, test_data_list = self.creat_oversampling_data_list()
        #建立訓練資料集
        self.train_data_size = train_data_list.shape[0]
        self.train_dataset, self.train_step_each_epoch = self.creat_jpg_dataset(train_data_list, self.train_data_size)
        self.train_dataset_iterator = iter(self.train_dataset)
        #建立測試資料集
        self.test_data_size = test_data_list.shape[0]
        self.test_dataset, self.test_step_each_epoch = self.creat_jpg_dataset(test_data_list, self.test_data_size)
        self.test_dataset_iterator = iter(self.test_dataset)

    #預計剩餘時間
    def ETA_time(self, start_time, end_time, step, epoch):     
        run_time = end_time -  start_time
        epoch = epoch + 1
        remainder_steps = (self.train_step_each_epoch - step) + ((self.epochs - epoch) * self.train_step_each_epoch)
        eta_time_seconds = math.floor(run_time * remainder_steps)
        if eta_time_seconds > 3600:
            eta_time_hours = math.floor(eta_time_seconds / 3600)
            eta_time_seconds = eta_time_seconds - (3600 * eta_time_hours)
            eta_time_minutes = math.floor(eta_time_seconds / 60)
            eta_time_seconds = eta_time_seconds - (60 * eta_time_minutes)
            output_str = f"預計剩餘時間: {eta_time_hours}小時 {eta_time_minutes}分鐘 {eta_time_seconds}秒"
        elif eta_time_seconds > 60:
            eta_time_minutes = math.floor(eta_time_seconds / 60)
            eta_time_seconds = eta_time_seconds - (60 * eta_time_minutes)
            output_str = f"預計剩餘時間: {eta_time_minutes}分鐘 {eta_time_seconds}秒"
        else:
            output_str = f"預計剩餘時間: {eta_time_seconds}秒"
            
        lenght = len(output_str) * 2
        print(output_str + "                                ", end = '\r')

    #模型訓練主體
    def train_loop(self):
        self.creat_dataset()
        self.creat_model()
        self.model.summary()

        for epoch in range(self.epochs):
            print(epoch)
            step = 0
            print_loss_value = 0
            
            while step < self.train_step_each_epoch:
                start_time = time.time()
                batch_file_label, batch_image = next(self.train_dataset_iterator)
                batch_file_label = tf.strings.to_number(batch_file_label, out_type = tf.dtypes.int32)
                batch_file_label = tf.one_hot(batch_file_label, self.number_of_class)

                self.loss_value = self.model_loss(batch_image, batch_file_label)
                print_loss_value += self.loss_value.numpy()
                step += 1
                end_time = time.time()
                self.ETA_time(start_time, end_time, step, epoch)

                if step % 500 == 0:
                    print_loss_value = print_loss_value / 500
                    print(f"loss value: {print_loss_value}")
                    print_loss_value = 0
            
            train_accuracy   = self.model_train_data_test()
            #current_accuracy = train_accuracy
            current_accuracy = self.model_test()
            if self.best_accuracy is not None:
                if current_accuracy >= self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    self.model.save("skin_cancer_classification")
            else:
                self.best_accuracy = current_accuracy
                self.model.save("skin_cancer_classification")

    #模型再訓練訓練
    def load_model_and_training(self):
        self.creat_dataset()
        self.model = tf.keras.models.load_model("skin_cancer_classification")

        self.best_accuracy = self.model_test()
        for epoch in range(self.epochs):
            print(epoch)
            step = 0
            print_loss_value = 0

            while step < self.train_step_each_epoch:
                start_time = time.time()
                batch_file_label, batch_image = next(self.train_dataset_iterator)
                batch_file_label = tf.strings.to_number(batch_file_label, out_type = tf.dtypes.int32)
                batch_file_label = tf.one_hot(batch_file_label, self.number_of_class)

                self.loss_value = self.model_loss(batch_image, batch_file_label)
                print_loss_value += self.loss_value.numpy()
                step += 1
                end_time = time.time()
                self.ETA_time(start_time, end_time, step, epoch)

                if step % 500 == 0:
                    print_loss_value = print_loss_value / 500
                    print(f"loss value: {print_loss_value}")
                    print_loss_value = 0

            train_accuracy   = self.model_train_data_test()
            #current_accuracy = train_accuracy
            current_accuracy = self.model_test()
            if self.best_accuracy is not None:
                if current_accuracy >= self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    self.model.save("skin_cancer_classification")
            else:
                self.best_accuracy = current_accuracy
                self.model.save("skin_cancer_classification")

    #驗證資料集模型測試
    def model_test(self):
        step = 0
        accuracy_list = []
        each_class_accuracy = [0.] * self.number_of_class

        while step < self.test_step_each_epoch:
                batch_file_label, batch_image = next(self.test_dataset_iterator)
                    
                batch_file_label         = tf.strings.to_number(batch_file_label, out_type = tf.dtypes.int32)
                batch_file_label_one_hot = tf.one_hot(batch_file_label, self.number_of_class)

                model_predict = self.model.predict(batch_image)
                accuracy      = tf.keras.metrics.categorical_accuracy(batch_file_label_one_hot, model_predict)
                for index in range(batch_file_label.shape[0]):
                    each_class_accuracy[batch_file_label.numpy()[index]] += accuracy.numpy()[index]
                accuracy      = tf.math.reduce_mean(accuracy)
                accuracy_list.append(accuracy.numpy())
                step += 1
                    
        output_accuracy = np.mean(accuracy_list)
        print(f"Test Accuracy: {output_accuracy}")
        for index in range(self.number_of_class):
            output_class_accuracy = each_class_accuracy[index] / self.test_data_size
            label = index + 1
            print(f"Class {label} Test accuracy: {output_class_accuracy}")

        return output_accuracy

    #訓練資料集模型測試
    def model_train_data_test(self):    
        step = 0
        accuracy_list = []
        each_class_accuracy = [0.] * self.number_of_class

        while step < self.train_step_each_epoch:
                batch_file_label, batch_image = next(self.train_dataset_iterator)
                    
                batch_file_label         = tf.strings.to_number(batch_file_label, out_type = tf.dtypes.int32)
                batch_file_label_one_hot = tf.one_hot(batch_file_label, self.number_of_class)

                model_predict = self.model.predict(batch_image)
                accuracy      = tf.keras.metrics.categorical_accuracy(batch_file_label_one_hot, model_predict)
                for index in range(batch_file_label.shape[0]):
                    each_class_accuracy[batch_file_label.numpy()[index]] += accuracy.numpy()[index]
                accuracy      = tf.math.reduce_mean(accuracy)
                accuracy_list.append(accuracy.numpy())
                step += 1
                    
        output_accuracy = np.mean(accuracy_list)
        print(f"Train data Accuracy: {output_accuracy}")
        for index in range(self.number_of_class):
            output_class_accuracy = each_class_accuracy[index] / self.train_data_size
            label = index + 1
            print(f"Class {label} Train accuracy: {output_class_accuracy}")

        return output_accuracy

    #載入模型並測試
    def load_model_and_test(self):
        self.creat_dataset()
        self.model = tf.keras.models.load_model("skin_cancer_classification")

        accuracy = self.model_test()
                    
    #計算損失值並更新權重
    @tf.function
    def model_loss(self, batch_image, batch_file_label):
        with tf.GradientTape() as tape:
            model_output = self.model(batch_image)
            loss_value   = self.loss_function(model_output, batch_file_label)
           
        #計算梯度
        weight = self.model.trainable_variables
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        self.optimizer.apply_gradients(zip(gradients, weight))
            
        return loss_value

skin_cancer_model = skin_cancer_classification_model(number_of_class, each_class_data_size, channels,
                                                     epochs, batch_size, learning_rate, dataset_buffer_size, data_size, data_path)

#模型訓練
skin_cancer_model.train_loop()
##模型再訓練
#skin_cancer_model.load_model_and_training()
##模型測試
#skin_cancer_model.load_model_and_test()