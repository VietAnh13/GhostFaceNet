import time
import evals
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skimage.io import imread
from face_detector import YoloV5FaceDetector
from datetime import datetime

# IMAGE FOLDER PATH
image_folder_path = r"./datasets/testcase/imgs/"

# CSV PATH
csv_path = r"./datasets/testcase/csv/pair_imgs.csv"

# LOAD MODEL
basic_model = keras.models.load_model('checkpoints/GN_W1.3_S2_ArcFace_epoch48.h5', compile=False)

start_time = time.time()

current_datetime = datetime.now()
sub_folder_name = current_datetime.strftime("%Y%m%d%H%M%S/")
dest_path = './crop_dataset/' + sub_folder_name
YoloV5FaceDetector().detect_in_folder_2(image_folder_path, dest_path)
print()

def read_csv_file(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    image_list = df.iloc[:, [0, 1]].values
    is_same_list = df.iloc[:, 2].values

    return image_list.flatten(), is_same_list

image_list, is_same_list = read_csv_file(csv_path)

for i in range(len(image_list)):
    image_list[i] = dest_path + image_list[i]

# BIN DATA PATH
bin_data_path = r"./binn/vastd.bin"
bb = [tf.image.encode_jpeg(imread(ii)).numpy() for ii in image_list]
with open(bin_data_path, "wb") as ff:
    pickle.dump([bb, image_list, is_same_list], ff)

ee = evals.eval_callback(basic_model, bin_data_path, batch_size=256, flip=True, PCA_acc=True)
ee.on_epoch_end(0)

end_time = time.time()
total_time = end_time - start_time
print('Total running time:', round(total_time, 5), '(s)')
