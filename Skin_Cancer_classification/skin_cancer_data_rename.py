import tensorflow as tf
import numpy as np
import concurrent.futures
import pathlib
import math
import os
from tensorflow.data import Dataset

#將影像的名稱加入類別標籤
def imagedata_rename_withdataset(image_id, label, batch_size,
                                 data_path, new_data_path, new_size):
    #建立dataste
    dataset, steps  = creat_jpg_dataset(image_id, data_path, image_id.shape[0], batch_size)
    dataset_iterator = iter(dataset)
    start_index = 0

    for index in range(steps):
        images, file_name  = dataset_iterator.get_next()
        #尋找輸入影像在原list中的位置
        file_type          = tf.convert_to_tensor([".jpg"] * image_id.shape[0])
        condition          = lambda image_id, file_name, index, image_index : index < file_name.shape[0]
        while_index        = 0
        image_index        = tf.constant([0], dtype = tf.int64)
        while_loop_return  = tf.while_loop(condition, find_image_id_index,
                                           (tf.strings.join([image_id, file_type]), file_name, while_index, image_index),
                                           parallel_iterations = batch_size)
        image_index        = while_loop_return[-1]
        image_index        = tf.split(image_index, [1, image_index.shape[0] - 1], 0)[-1]
        #變更影像大小
        images             = tf.map_fn(fn = tensor_image_resize,elems = images)
        #建立包含類別的影像名稱並輸出
        image_label        = tf.convert_to_tensor(label.numpy()[image_index.numpy()])
        image_label        = tf.strings.as_string(image_label)
        image_name         = tf.convert_to_tensor(image_id.numpy()[image_index.numpy()])
        new_file_name      = tf.strings.join([image_name, image_label], "_")
        file_type          = tf.convert_to_tensor([".jpg"] * images.shape[0])
        new_file_name      = tf.strings.join([new_file_name, file_type])
        new_data_path_list = tf.convert_to_tensor([new_data_path] *images.shape[0])
        new_file_name      = tf.strings.join([new_data_path_list, new_file_name])
        condition          = lambda images, new_file_name, index : index < images.shape[0]
        while_index        = 0
        while_loop_return  = tf.while_loop(condition, write_image, (images, new_file_name, while_index),
                                           parallel_iterations = batch_size)
        

#由於tf dataset會重新排序給予的名單
#所以需要尋找輸入影像在原list中的位置
def find_image_id_index(image_id, file_name, index, image_index):
    file_index = tf.math.equal(image_id, file_name[index])
    file_index = tf.where(file_index)
    file_index = tf.squeeze(file_index)
    file_index = tf.expand_dims(file_index, 0)
    index += 1
    image_index = tf.concat([image_index, file_index], 0)
    return (image_id, file_name, index, image_index)
#匯出影像
def write_image(images, new_file_name, index):
    image = tf.io.encode_jpeg(image = images[index], quality = 100)
    tf.io.write_file(new_file_name[index], image)
    index += 1
    return (images, new_file_name, index)
#改變影像大小
def tensor_image_resize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, tf.convert_to_tensor(new_size))
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return image

#建立jpg影像dataset並回傳要幾次才能跑完
def creat_jpg_dataset(data_id, data_path, data_size, batch_size):
    file_name = tf.strings.join([tf.convert_to_tensor(data_id),
                                tf.convert_to_tensor([".jpg"] * len(data_id))])
    data_path_list = tf.strings.join([tf.convert_to_tensor([data_path] * len(data_id)),
                                file_name])
    #data_list = Dataset.list_files(data_path_list, shuffle = False)
    dataset   = Dataset.from_tensor_slices(data_path_list)
    dataset   = dataset.map(path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)    
    dataset   = dataset.batch(batch_size)
    steps     = math.ceil(data_size / batch_size)

    return dataset, steps

#取得檔案名稱
def get_file_name(file_path):
    split_string = tf.strings.split(file_path, "/")
    return split_string[-1]

#dataset 處理函式
def path_process(file_path):
    #匯入檔案
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    file_name = get_file_name(file_path)
    return image, file_name

#將image傳成tensor
def decode_image(image):
    #將image轉成nuit8 channel為3的tensor
    image = tf.io.decode_image(image, channels = 3)
    return image

#try:
#    #取得實體GPU數量
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    if gpus:
#        for gpu in gpus:
#        #將GPU記憶體使用率設為動態成長
#        #有建立虛擬GPU時不可使用
#        #tf.config.experimental.set_memory_growth(gpu, True)
#        #建立虛擬GPU
#            tf.config.experimental.set_virtual_device_configuration(
#                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 500)])
#except Exception as e:
#    print(e)

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
    
#number_of_class          = tf.unique(patient_data[2]).y.shape[0]
label                    = tf.unique(patient_data[2]).idx
image_id                 = patient_data[1]
original_data_path = "E:/GitHub Program/skin_cancer_data/HAM10000_images/"
new_data_path      = "E:/GitHub Program/skin_cancer_data/HAM10000_images_rename"

number_of_process       = 2
number_of_data_to_split = 1
new_size                = (200,200)
batch_size              = 500

run_rename_function = True
#將原始資料的檔案名稱加入類別標籤以便訓練
try:
    os.makedirs(new_data_path)
except FileExistsError:
    print("Folder Exist")
    imagedata_rename_withdataset(image_id, label,  batch_size, original_data_path, new_data_path + "/", new_size)
    