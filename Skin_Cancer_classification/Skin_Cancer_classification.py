import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.data import Dataset
import math
import pathlib
import os

#dataset 處理函式
def path_process(file_path):
    file_name = get_file_name(file_path)
    #匯入檔案
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return file_name, image

#取得檔案名稱
def get_file_name(file_path):
    #以python內建的檔案路徑分隔符號為標準分割字串
    split_string = tf.strings.split(file_path, os.path.sep)
    #回傳檔名
    return split_string[-1]

#將image傳成tensor
def decode_image(image):
    #將image轉成nuit8 channel為3的tensor
    image = tf.io.decode_image(image, channels = 3)
    #將image轉成指定的data type，並且作正規化
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

#建立jpg影像dataset並回傳要幾次才能跑完一個epoch
def creat_jpg_dataset(data_path, data_size, batch_size, dataset_buffer_size):
    data_list = Dataset.list_files(data_path + "*.jpg", shuffle = False)
    dataset   = data_list.map(path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset   = dataset.shuffle(dataset_buffer_size).repeat()
    dataset   = dataset.batch(batch_size)
    step_each_epoch = math.ceil(data_size / batch_size)

    return dataset, step_each_epoch

#建立分類模型
def creat_model(channels, number_of_class):
    input_layer   = Input(shape = [None, None, channels, ])
    con2d_layer   = layers.Conv2D(16, 3)(input_layer)
    act_layer     = layers.PReLU()(con2d_layer)
    bn_layer      = layers.BatchNormalization()(act_layer)
    dropout_layer = layers.Dropout(0.5)(bn_layer)
    con2d_layer   = layers.Conv2D(32, 3)(dropout_layer)
    act_layer     = layers.PReLU()(con2d_layer)
    bn_layer      = layers.BatchNormalization()(act_layer)
    dropout_layer = layers.Dropout(0.5)(bn_layer)
    con2d_layer   = layers.Conv2D(64, 3)(dropout_layer)
    act_layer     = layers.PReLU()(con2d_layer)
    bn_layer      = layers.BatchNormalization()(act_layer)
    dropout_layer = layers.Dropout(0.5)(bn_layer)
    con2d_layer   = layers.Conv2D(128, 3)(dropout_layer)
    act_layer     = layers.PReLU()(con2d_layer)
    bn_layer      = layers.BatchNormalization()(act_layer)
    dropout_layer = layers.Dropout(0.5)(bn_layer)
    flatten_layer = layers.Flatten()(dropout_layer)
    hidden_layer  = layers.Dense(20)(flatten_layer)
    act_layer     = layers.PReLU()(hidden_layer)
    dropout_layer = layers.Dropout(0.5)(act_layer)
    hidden_layer  = layers.Dense(40)(dropout_layer)
    act_layer     = layers.PReLU()(hidden_layer)
    dropout_layer = layers.Dropout(0.5)(act_layer)
    hidden_layer  = layers.Dense(40)(dropout_layer)
    act_layer     = layers.PReLU()(hidden_layer)
    dropout_layer = layers.Dropout(0.5)(act_layer)
    output_layer  = layers.Dense(number_of_class, activation = 'softmax')(dropout_layer)

    return Model(input_layer, output_layer)

#優化器設定
optimizer = tf.keras.optimizers.Adam()

#超參數設定
channels            = 3
epochs              = 500
batch_size          = 32
dataset_buffer_size = 200
train_data_size     = 5000
test_data_size      = 5015
train_data_path     = "D:/皮膚癌資料/HAM10000_images_part_1/"
test_data_path      = "D:\皮膚癌資料\HAM10000_images_part_2"

##建立訓練資料集
#train_dataset, train_step_each_epoch = creat_jpg_dataset(train_data_path, train_data_size,
#                                                         batch_size, dataset_buffer_size)
##建立測試資料集
#test_dataset, test_step_each_epoch   = creat_jpg_dataset(test_data_path, test_data_size,
#                                                         batch_size, dataset_buffer_size)

#匯入類別資料
patient_data_path = "D:/皮膚癌資料/HAM10000_metadata.csv"
patient_data  = pathlib.Path(patient_data_path).read_text()
#以分隔符號將資料分割並移除header
patient_data  = patient_data.split("\n")[1:]
#定義每個特徵的資料類別
col_data_type = [str(), str(), str(), str(), float(), str(), str()]
#根據給予的資料類別建立一個包含數個tensor的list
#每個tensor代表一個特徵
patient_data  = tf.io.decode_csv(patient_data, record_defaults = col_data_type)

number_of_class = tf.unique(patient_data[2]).y.shape[0]
label           = tf.unique(patient_data[2]).idx
image_id        = patient_data[1]
classification_model = creat_model(channels, number_of_class)

def train_loop(classification_model, train_dataset, train_step_each_epoch, label, image_id, epochs):
    train_dataset_iterator = iter(train_dataset)
    for epoch in range(epochs):
        step = 0
        while step < train_step_each_epoch:
            batch_file_name, batch_image = next(train_dataset_iterator)
            a = tf.math.equal(image_id, batch_file_name)
            step += 1
    
train_loop(classification_model, train_dataset, train_step_each_epoch, label, image_id, epochs)