import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.data import Dataset

#建立image dataset
train_data_path = "D:/皮膚癌資料/HAM10000_images_part_1/"
test_data_path = "D:\皮膚癌資料\HAM10000_images_part_2"

#dataset 處理函式
def path_process(file_path):
    file_label = get_file_label(file_path)
    #匯入檔案
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return file_label, image

#取得檔案名稱
def get_file_label(file_path):
    #以python內建的檔案路徑分隔符號為標準分割字串
    split_string = tf.strings.split(file_path, os.path.sep)
    #回傳檔名
    return split_string[-2]

#將image傳成tensor
def decode_image(image):
    #將image轉成nuit8 channel為3的tensor
    image = tf.io.decode_image(image, channels = 3)
    #將image轉成指定的data type，並且作正規化
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

#建立訓練資料集
train_data_list = Dataset.list_files(train_data_path + "*.jpg")
train_data_set  = train_data_list.map(path_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)