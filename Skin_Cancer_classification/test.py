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

try:
    #取得實體GPU數量
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
        #將GPU記憶體使用率設為動態成長
        #有建立虛擬GPU時不可使用
        #tf.config.experimental.set_memory_growth(gpu, True)
        #建立虛擬GPU
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 100)])
except Exception as e:
    print(e)

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
    
patient_id_not_repeat = tf.unique_with_counts(patient_data[0]).count.numpy()
patient_id_not_repeat = tf.math.equal(patient_id_not_repeat, 1)
patient_id_not_repeat = tf.where(patient_id_not_repeat)
image_id             = patient_data[1]

number_of_patient_id_not_repeat  = patient_id_not_repeat.shape[0]
number_of_patient_id_repeat = image_id.shape[0] - number_of_patient_id_not_repeat
print(f"number_of_patient_id_repeat:{number_of_patient_id_repeat}")
print(f"number_of_patient_id_not_repeat:{number_of_patient_id_not_repeat}")