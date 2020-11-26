import tensorflow as tf
import numpy as np
import pathlib
import math
from tensorflow.data import Dataset

def unbalance_data_oversampling(class_data_list, steps, batch_size, data_path, class_list, rotation_range,
                                zoom_rate, width_shift_rate, height_shift_rate, patient_data, file_name_start_index):
    dataset          = creat_jpg_dataset(class_data_list, batch_size)
    dataset_iterator = iter(dataset)
    
    for step in range(steps):
        images, file_labels  = dataset_iterator.get_next()
        new_images           = np.zeros(images.shape) 

        condition          = lambda images, new_images, rotation_range, zoom_rate, width_shift_rate, height_shift_rate, index : index < images.shape[0]
        while_index        = 0
        while_loop_return  = tf.while_loop(condition, image_process,
                                           (images, new_images, rotation_range, zoom_rate, width_shift_rate, height_shift_rate, while_index),
                                           parallel_iterations = batch_size)
        images = tf.convert_to_tensor(while_loop_return[1])
        new_file_name   = ["oversampling"] * batch_size
        file_name_count = tf.constant(range(file_name_start_index, file_name_start_index + batch_size)) 
        new_file_name   = tf.strings.join([new_file_name, tf.strings.as_string(file_name_count)], "_")
        for index in range(len(patient_data)):
            if index == 1:
                patient_data[index] = tf.concat([patient_data[index], new_file_name], 0)
            elif index == 2:
                condition          = lambda file_labels, class_list, index : index < file_labels.shape[0]
                while_index        = 0
                while_loop_return  = tf.while_loop(condition, file_labels_changes, (file_labels.numpy(), class_list, while_index),
                                                   parallel_iterations = batch_size)
                patient_data[index] = tf.concat([patient_data[index], tf.convert_to_tensor(while_loop_return[0])], 0)
            elif index == 4:
                empty_list = [0.] * batch_size
                patient_data[index] = tf.concat([patient_data[index], empty_list], 0)
            else:
                empty_list = [""] * batch_size
                patient_data[index] = tf.concat([patient_data[index], empty_list], 0)

        new_file_name   = tf.strings.join([new_file_name, file_labels], "_")
        data_type_list  = [".jpg"] * batch_size
        new_file_name   = tf.strings.join([new_file_name, data_type_list])
        data_path_list  = [data_path] * batch_size
        new_file_name   = tf.strings.join([data_path_list, new_file_name])

        condition          = lambda images, new_file_name, index : index < images.shape[0]
        while_index        = 0
        while_loop_return  = tf.while_loop(condition, write_image, (images, new_file_name, while_index),
                                           parallel_iterations = batch_size)
        file_name_start_index += batch_size

    return patient_data, file_name_start_index



#建立jpg影像dataset並回傳要幾次才能跑完
def creat_jpg_dataset(class_data_list, batch_size):
    dataset   = Dataset.from_tensor_slices(class_data_list)
    dataset   = dataset.map(path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)    
    dataset   = dataset.shuffle(class_data_list.shape[0], reshuffle_each_iteration=True).repeat()
    dataset   = dataset.batch(batch_size)
    
    return dataset

#取得檔案類別
def get_file_label(file_path):
    split_string = tf.strings.split(file_path, "/")
    label        = tf.strings.split(split_string[-1], "_")
    label        = tf.strings.split(label[-1], ".")

    return label[0]

#dataset 處理函式
def path_process(file_path):
    #匯入檔案
    image      = tf.io.read_file(file_path)
    image      = decode_image(image)
    file_label = get_file_label(file_path)
    return image, file_label

#將image傳成tensor
def decode_image(image):
    #將image轉成nuit8 channel為3的tensor
    image = tf.io.decode_image(image, channels = 3)
    #將image轉成指定的data type，並且作正規化
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

#影像處理函式
def image_process(images, new_images, rotation_range, zoom_rate, width_shift_rate, height_shift_rate, index):
    
    image = images[index].numpy()
    image = tf.keras.preprocessing.image.random_rotation(image, rotation_range, row_axis = 0, col_axis = 1, channel_axis = 2)
    image = tf.keras.preprocessing.image.random_zoom(image, zoom_rate, row_axis = 0, col_axis = 1, channel_axis = 2)
    image = tf.keras.preprocessing.image.random_shift(image, width_shift_rate, height_shift_rate, row_axis = 0, col_axis = 1, channel_axis = 2)
    new_images[index] = image

    index +=1

    return (images, new_images, rotation_range, zoom_rate, width_shift_rate, height_shift_rate, index)

#file_labels轉換
def file_labels_changes(file_labels, class_list, index):
    file_labels[index] = class_list[int(file_labels[index])].numpy()
    index += 1

    return (file_labels, class_list, index)

#匯出影像
def write_image(images, new_file_name, index):
    image = tf.image.convert_image_dtype(images[index], tf.uint8)
    image = tf.io.encode_jpeg(image = image, quality = 100)
    tf.io.write_file(new_file_name[index], image)
    index += 1
    return (images, new_file_name, index)

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
    
number_of_class   = tf.unique(patient_data[2]).y.shape[0]
class_list        = tf.unique(patient_data[2]).y
label             = tf.unique(patient_data[2]).idx
each_class_number = tf.unique_with_counts(patient_data[2]).count.numpy()
image_id          = patient_data[1]
data_path         = "E:/GitHub Program/skin_cancer_data/HAM10000_images_rename/"
#隨機旋轉範圍(以度為單位)
rotation_range    = 40
#隨機縮放比例(須為tuple的float，分別對應寬和高)
zoom_rate         = (0.5, 1.5)
#隨機位移比例
width_shift_rate      = 0.2
height_shift_rate     = 0.2
batch_size            = 500
file_name_start_index = 0

maximun_of_class_data = np.max(each_class_number)
for index in range(number_of_class):
    #判斷當前類別有的資料數量是否小於最大值
    if each_class_number[index] < maximun_of_class_data:
        #計算要產生多少筆資料
        over_sample_times = maximun_of_class_data - each_class_number[index]
        steps             = math.ceil(over_sample_times / batch_size)

        class_data_index = tf.math.equal(label, index)
        class_data_index = tf.where(class_data_index)
        class_data_index = tf.squeeze(class_data_index)
        class_data_id    = tf.convert_to_tensor(image_id.numpy()[class_data_index], tf.dtypes.string)
        class_data_label = tf.convert_to_tensor(label.numpy()[class_data_index]) 
        class_data_label = tf.strings.as_string(class_data_label)
        class_data_id    = tf.strings.join([class_data_id, class_data_label], "_")
        image_type       = ['.jpg'] * class_data_id.shape[0]
        class_data_id    = tf.strings.join([class_data_id, image_type])
        image_path       = [data_path] * class_data_id.shape[0]
        class_data_list  = tf.strings.join([image_path, class_data_id])

        patient_data, file_name_start_index = unbalance_data_oversampling(class_data_list, steps, batch_size, data_path, class_list,
                                                                          rotation_range, zoom_rate, width_shift_rate, height_shift_rate,
                                                                          patient_data, file_name_start_index)

#匯出csv檔
patient_data_path = "E:/GitHub Program/skin_cancer_data/HAM10000_metadata_oversample.csv"
patient_data[4] = tf.strings.as_string(patient_data[4])
csv_header = ["lesion_id","image_id","dx","dx_type","age","sex","localization"]

for index in range(len(patient_data)):
    joint_header = tf.convert_to_tensor(csv_header[index])
    joint_header = tf.reshape(joint_header, [1,])
    patient_data[index] = tf.concat([joint_header, patient_data[index]], 0)
    #patient_data[index] = patient_data[index].numpy()
patient_data = tf.convert_to_tensor(patient_data)
patient_data = tf.transpose(patient_data).numpy()
patient_data = patient_data.astype(str)
    
np.savetxt(fname = patient_data_path, X = patient_data, fmt = '%s', delimiter = ',')
