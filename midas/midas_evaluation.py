import cv2
import glob
import os
import torch
from midas.StereoGeneratorEngine import StereoGeneratorEngine as sge
from tqdm import tqdm
import statistics

my_path_l = "d:\\dane\\kitti\\testing\\image_2\\*.png"
my_path_r = "d:\\dane\\kitti\\testing\\image_3\\"

files = glob.glob(my_path_l)
print(len(files))
print(files)


theta = 0.75

#[midas, device, transform] = sge.prepare_dnn("DPT_Hybrid")
#[midas, device, transform] = sge.prepare_dnn("DPT_Large")
[midas, device, transform] = sge.prepare_dnn("MiDaS_small")

def calculate_mae(ri, right_img):
    mae = 0
    for x in range(ri.shape[0]):
        for y in range(ri.shape[1]):
            for z in range(ri.shape[2]):
                #mae = mae + abs(int(li[x,y,z]) - int(ri[x,y,z]))
                mae = mae + abs(int(right_img[x, y, z]) - int(ri[x, y, z]))
    mae = mae / (ri.shape[0] * ri.shape[1] * ri.shape[2])
    return(mae)

all_mae = []

all_resolution = [(640, 360),(1280, 720)]
all_maxdeviation = [25,50,75]

for resolution in all_resolution:
    for maxdeviation in all_maxdeviation:
        for a in tqdm(range(len(files))):
            bn = os.path.basename(files[a])
            li = cv2.imread(files[a])
            ri = cv2.imread(my_path_r + bn)

            li = cv2.resize(li, resolution)
            ri = cv2.resize(ri, resolution)

            frame = li
            depth_old = None
            (right_img, depth_old, output) = sge.generate_right(frame, device, depth_old, midas, transform,
                                                            theta, maxdeviation)

            ri = ri[0:ri.shape[0], 0:ri.shape[1]-maxdeviation, 0:ri.shape[2]]
            right_img = right_img[0:right_img.shape[0], 0:right_img.shape[1]-maxdeviation, 0:right_img.shape[2]]

            #cv2.imshow("right_imgii", ri)
            #cv2.imshow("right_img", right_img)
            #cv2.waitKey()
            mae= calculate_mae(ri, right_img)
            all_mae.append(mae)

        mean_ = sum(all_mae)/len(all_mae)
        stdev_ = statistics.stdev(all_mae)
        print("" + str(resolution) + "," + str(maxdeviation) + "," + str(mean_) + "," + str(stdev_) + "\n")

"""
mae = 0
for x in range(li.shape[0]):
    for y in range(li.shape[1]):
        for z in range(li.shape[2]):
            mae = mae + abs(int(li[x,y,z]) - int(ri[x,y,z]))
mae = mae / (li.shape[0] * li.shape[1] * li.shape[2])
#cv2.waitKey()
print(mae)
"""