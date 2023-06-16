import cv2
import glob
import os
from StereoGeneratorEngine import StereoGeneratorEngine as sge
from tqdm import tqdm
import statistics

my_path_l = "d:\\dane\\kitti\\testing\\image_2\\*.png"
my_path_r = "d:\\dane\\kitti\\testing\\image_3\\"

files = glob.glob(my_path_l)
print(len(files))
print(files)


theta = 0.75

from keras.models import load_model
from StereoGeneratorEngine.layers import BilinearUpSampling2D
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

model_file = "modelOK_T_NYU.h5"

model = load_model(model_file, custom_objects=custom_objects, compile=False)

def calculate_mae(ri, right_img):
    mae = 0
    for x in range(ri.shape[0]):
        for y in range(ri.shape[1]):
            for z in range(ri.shape[2]):
                mae = mae + abs(int(right_img[x, y, z]) - int(ri[x, y, z]))
    mae = mae / (ri.shape[0] * ri.shape[1] * ri.shape[2])
    return(mae)

all_mae = []

all_maxdeviation = [25,50,75]
all_resolution = [(640, 360), (1280, 720)]

for resolution in all_resolution:
    for maxdeviation in all_maxdeviation:
        for a in tqdm(range(0,len(files))):
            bn = os.path.basename(files[a])
            li = cv2.imread(files[a])
            ri = cv2.imread(my_path_r + bn)

            li = cv2.resize(li, resolution)
            ri = cv2.resize(ri, resolution)

            frame = li
            depth_old = None

            (right_img, depth_old, output) = sge.generate_right(frame, model, depth_old,
                                                            theta, maxdeviation)

            ri = ri[0:ri.shape[0], 0:ri.shape[1]-maxdeviation, 0:ri.shape[2]]
            right_img = right_img[0:right_img.shape[0], 0:right_img.shape[1]-maxdeviation, 0:right_img.shape[2]]
            mae= calculate_mae(ri, right_img)
            all_mae.append(mae)

        mean_ = sum(all_mae)/len(all_mae)
        stdev_ = statistics.stdev(all_mae)
        print("" + str(resolution) + "," + str(maxdeviation) + "," + str(mean_) + "," + str(stdev_) + "\n")