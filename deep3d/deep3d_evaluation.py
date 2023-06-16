import os
import argparse
from tqdm import tqdm

import numpy as np
import torch

from data import transform,impro
from utils import util,ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int,help="choose your device")
parser.add_argument("--model", default='./export/deep3d_v1.0.pt', type=str,help="input model path")
parser.add_argument("--video", default='./medias/wood.mp4', type=str,help="input video path")
parser.add_argument("--out", default='./results/wood.mp4', type=str,help="output video path")
parser.add_argument('--inv', action='store_true', help='some video need to reverse left and right views')
parser.add_argument("--tmpdir", default='./tmp', type=str,help="output video path")
opt = parser.parse_args()

opt.model = 'export/deep3d_v1.0_640x360_cuda.pt'
#opt.model = 'export/deep3d_v1.0_1280x720_cuda.pt'

my_file = ['banka.mp4','biblioteka.mp4','dolina.mp4','ekspres.mp4','kolo.mp4','kwiatki.mp4','lotnisko.mp4','lwice.mp4',
'mycie_rak.mp4','owce.mp4','papuga.mp4','porsche.mp4','port.mp4','ptaki.mp4','rowerzysta.mp4','seoul.mp4','szlak.mp4',
'telefon.mp4','wiatrak.mp4','wodospad.mp4','wulkan.mp4','zaba.mp4','zegarek.mp4']
file_id = 0
opt.video = 'd:/data/video/' + my_file[file_id]
#opt.out = 'd:/data/video/deep3d_640x360/' + my_file[file_id]
opt.out = 'd:/data/video/deep3d_1280x720/' + my_file[file_id]


net = torch.jit.load(opt.model)
net.eval()
process = transform.PreProcess()

if 'cuda' in opt.model and torch.cuda.is_available():
    net.to(opt.gpu_id).half()
    process.to(opt.gpu_id).half()
else:
    opt.gpu_id = -1

out_width  = int(os.path.basename(opt.model).split('_')[2].split('x')[0])
out_height = int(os.path.basename(opt.model).split('_')[2].split('x')[1])

fps,duration,height,width = ffmpeg.get_video_infos(opt.video)
video_length = int(fps*duration)

util.clean_tempfiles(opt.tmpdir)
util.makedirs(os.path.split(opt.out)[0])
ffmpeg.video2voice(opt.video,os.path.join(opt.tmpdir, 'tmp.wav'))

tips_l = []
tips_r = []

import cv2
import glob


my_path_r = "d:\\dane\\kitti\\testing\\image_2\\"
my_path_l = "d:\\dane\\kitti\\testing\\image_3\\*.png"

files = glob.glob(my_path_l)
print(len(files))
print(files)


theta = 0.75


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

########################################################
id_cap = 0
alpha = 5
cap = cv2.VideoCapture(opt.video)
frames_pool = []
frames_pool_right = []

output = np.zeros((out_height * 1, out_width * 2, 3), np.uint8)
for i in range(alpha * 2 + 1):
    #ret, cur_frame = cap.read()

    ########################################################
    bn = os.path.basename(files[id_cap])
    cur_frame = li = cv2.imread(files[id_cap])
    ri = cv2.imread(my_path_r + bn)
    id_cap = id_cap + 1
    ########################################################

    if height != out_height or width != out_width:
        cur_frame = cv2.resize(cur_frame, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
    frames_pool.append(torch.from_numpy(cur_frame))
    ########################################################
    frames_pool_right.append(ri)

x0 = frames_pool[0]
if opt.gpu_id >= 0:
    x0 = x0.to(opt.gpu_id).half()
x0 = process(x0)

print("start inference...")
idd = 0
all_mae = []


ssss = range(video_length)

for frame in tqdm(range(video_length)):
    bn = os.path.basename(files[idd])
    cur_frame_right = cv2.imread(my_path_r + bn)
    cur_frame_right = cv2.resize(cur_frame_right, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
    idd = idd + 1

    if frame < alpha:
        beta = 0
    elif alpha <= frame < video_length - alpha:
        beta = -(frame - alpha)

    if alpha < frame < video_length - alpha:
        #ret, cur_frame = cap.read()
        ########################################################
        if id_cap >= len(files):
            break

        bn = os.path.basename(files[id_cap])
        cur_frame = li = cv2.imread(files[id_cap])
        ret = True
        ri = cv2.imread(my_path_r + bn)
        id_cap = id_cap + 1
        ########################################################



        if height != out_height or width != out_width:
            cur_frame = cv2.resize(cur_frame, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)

        if not ret or cur_frame is None:
            break
        frames_pool.pop(0)
        frames_pool.append(torch.from_numpy(cur_frame))

    x1 = frames_pool[np.clip(frame - alpha + beta, 0, alpha * 2)]
    x2 = frames_pool[np.clip(frame - 1 + beta, 0, alpha * 2)]
    x3 = frames_pool[frame + beta]
    x4 = frames_pool[np.clip(frame + 1 + beta, 0, alpha * 2)]
    x5 = frames_pool[np.clip(frame + alpha + beta, 0, alpha * 2)]

    if opt.gpu_id >= 0:
        x1, x2, x3, x4, x5 = x1.to(opt.gpu_id).half(), x2.to(opt.gpu_id).half(), x3.to(opt.gpu_id).half(), x4.to(
            opt.gpu_id).half(), x5.to(opt.gpu_id).half()
    x1, x2, x3, x4, x5 = process(x1), process(x2), process(x3), process(x4), process(x5)

    input_data = torch.cat((x1, x2, x0, x3, x4, x5), dim=0)
    input_data = input_data.reshape(1, *input_data.shape)

    with torch.no_grad():
        out = net(input_data)
        x0 = out.clone().detach()[0]

    left = x3
    right = out[0]

    right = transform.tensor2im(right)
    all_mae.append(calculate_mae(right, cur_frame_right))

import statistics
mean_ = sum(all_mae) / len(all_mae)
stdev_ = statistics.stdev(all_mae)
print("" + str((out_width, out_height)) + "," + str(mean_) + "," + str(stdev_) + "\n")
