import time
import evals
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skimage.io import imread
from face_detector import YoloV5FaceDetector
from datetime import datetime

def read_csv_file(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', header=None)
    image_list = df.iloc[:, [0, 1]].values
    is_same_list = df.iloc[:, 2].values

    return image_list.flatten(), is_same_list

def zip_step(imgs_path, csv_path, model_path, bin_path):
    # LOAD MODEL
    basic_model = keras.models.load_model(model_path, compile=False)

    # start_time = time.time()

    current_datetime = datetime.now()
    sub_folder_name = current_datetime.strftime("%Y%m%d%H%M%S/")
    dest_path = './crop_dataset/' + sub_folder_name
    YoloV5FaceDetector().detect_in_folder_2(imgs_path, dest_path)
    print()

    image_list, is_same_list = read_csv_file(csv_path)
    # print(len(is_same_list))
    # print(len(image_list))

    for i in range(len(image_list)):
        image_list[i] = dest_path + image_list[i]

    bb = [tf.image.encode_jpeg(imread(ii)).numpy() for ii in image_list]
    with open(bin_path, "wb") as ff:
        pickle.dump([bb, image_list, is_same_list], ff)

    for i in range(10):
        start_time = time.time()
        ee = evals.eval_callback(basic_model, bin_path, batch_size=256, flip=True, PCA_acc=True)
        ee.on_epoch_end(0)
        end_time = time.time()
        total_time = end_time - start_time
        print('Total running time:', round(total_time, 5), '(s)')
        fps = 1 / (total_time / len(is_same_list))
        print('FPS:', round(fps, 2))

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-d", "--data_path", type=str, default=None, help="Data path, containing images in folders")
    # parser.add_argument("-c", "--csv_path", type=str, default=None, help="CSV file, containing pair images")
    # parser.add_argument("-m", "--model_file", type=str, default=None, help="Model file, keras h5")
    # parser.add_argument("-B", "--save_bins", type=str, default=None, help="Save evaluating pair bin")

    parser.add_argument("-d", "--data_path", type=str, default='./datasets/testcase/imgs/', help="Data path, containing images in folders")
    parser.add_argument("-c", "--csv_path", type=str, default='./datasets/testcase/csv/pair_imgs.csv', help="CSV file, containing pair images")
    parser.add_argument("-m", "--model_file", type=str, default='./checkpoints/GN_W1.3_S2_ArcFace_epoch48.h5', help="Model file, keras h5")
    parser.add_argument("-B", "--save_bins", type=str, default='./binn/vastd.bin', help="Save evaluating pair bin")

    args = parser.parse_known_args(sys.argv[1:])[0]

    zip_step(args.data_path, args.csv_path, args.model_file, args.save_bins)

# python zip_step.py -d ./datasets/testcase/imgs/ -c ./datasets/testcase/csv/pair_imgs.csv -m ./checkpoints/GN_W1.3_S2_ArcFace_epoch48.h5 -B ./binn/vastd.bin