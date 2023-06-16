from numba import jit,njit, prange
import numpy as np
import cv2
#import torch
#from midas.StereoGeneratorEngine import Utils as sgeutils
import timeit
from tqdm import tqdm
import time

def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
#@njit(parallel=True)
def calculate_right(h, w, depth, deviation, left_image, right): # Function is compiled to machine code when called the first time
    for row in range(h):
        for col in range(w):
            #col_r = col - int((1 - depth[row][col]) * deviation)
            col_r = col - int(depth[row][col] * deviation)
            #if depth[row][col] > 0:
            #    col_r = col - int(1.0 / (depth[row][col]))
            if col_r >= 0:
                right[row][col_r] = left_image[row][col]
    return right

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
#@njit(parallel=True)
def mask_to_image(rows, cols, mask): # Function is compiled to machine code when called the first time
    ll = len(rows)
    for i in range(ll):
        mask[rows[i], cols[i]] = 255
    return mask


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def inpaint_fast_iter(mask, img):
    one_change = False

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
        #for y in range(mask.shape[1]-1, 0, -1):
            if mask[x, y] > 0:
                valueR = 0
                valueG = 0
                valueB = 0
                count = 0

                #for xx in [-1, 0, 1]:
                #    for yy in [-1,0,  1]:
                for xx in [-3, -2, -1, 0, 1, 2, 3]:
                    for yy in [-3, -2, -1, 0 ,1, 2, 3]:
                        if x + xx > 0 and y + yy > 0 and x + xx < mask.shape[0] and y + yy < mask.shape[1]:
                            if mask[x + xx, y + yy] == 0:
                                valueR = valueR + (img[x + xx, y + yy, 0])
                                valueG = valueG + (img[x + xx, y + yy, 1])
                                valueB = valueB + (img[x + xx, y + yy, 2])
                                count = count + 1
                if count > 0:
                    one_change = True
                    #rrrr = valueR / count
                    img[x, y, 0] = valueR / count
                    img[x, y, 1] = valueG / count
                    img[x, y, 2] = valueB / count
                    mask[x, y] = 0
    return one_change

def inpaint_fast(img, mask):
    img = np.copy(img)
    mask = np.copy(mask)
    one_change = True
    while one_change:
        one_change = inpaint_fast_iter(mask, img)
    return img

#EFFICIENT DEPTH IMAGE BASED RENDERING WITH EDGE DEPENDENT DEPTH FILTER AND INTERPOLATION
#VISUAL PERTINENT 2D-TO-3D VIDEO CONVERSION BY MULTI-CUE FUSION
def generate_stereo(left_image, depth, maxdeviation = 25):
    h, w, c = left_image.shape
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.full(left_image.shape, -1)
    right = calculate_right(h, w, depth[:,:,0], maxdeviation, left_image, right)
    right_fix = np.array(right)
    rows, cols = np.where(right[:,:,0] == -1)

    right_fix = right_fix.astype(np.uint8)

    mask = np.zeros((right.shape[0], right.shape[1]), dtype=np.uint8)
    mask = mask_to_image(rows, cols, mask)
    #https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/
    #https://www.geeksforgeeks.org/image-inpainting-using-opencv/

    # right_fix = inpaint_fast(right_fix, mask)
    # right_fix = cv2.inpaint(right_fix, mask, 2, cv2.INPAINT_NS)
    right_fix = cv2.inpaint(right_fix, mask, 2, cv2.INPAINT_TELEA)
    return right_fix

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def generate_right(frame, model, depth_old,
                   theta, maxdeviation):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    """
    #input_batch = transform(img).to(device)

    #start = time.time()
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    """

    frame_to_process = np.copy(frame)
    frame_to_process = cv2.resize(frame_to_process, (frame_to_process.shape[1] * 2, frame_to_process.shape[0] * 2))
    x = np.clip(frame_to_process / 255, 0, 1)
    inputs = np.expand_dims(x, 0)
    outputs = predict(model, inputs)
    # RGB and depth output
    img = np.copy(outputs[0, :, :, :])
    output = img
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    #img = (255 * img).astype(np.uint8)


    if depth_old is None:
        depth_old = output
    else:
        output = (theta * output) + ((1 - theta) * depth_old)
        depth_old = output

    # rescaled = 255.0 * ((np.copy(output) - np.min(output))/np.max(output))
    #rescaled = 255.0 * (output - np.min(output)) / (np.max(output) - np.min(output))
    #rescaled = rescaled.astype(np.uint8)

    ##############################
    # depth_map = write_depth(output, bits=2, reverse=False)
    depth_map = output
    left_img = frame

    right_img = generate_stereo(left_img, depth_map, maxdeviation=maxdeviation)
    return (right_img,depth_old, output)


def generate_stereo_from_VideoCapture(
        model,
        horizontal = False,
        include_sound = True,
        show_depth = True,
        show_frames = True,
        show_fps = True,
        resolution = (640, 360),
        theta = 0.75,
        maxdeviation = 25,
        in_path = None, out_path = None, out_path_sound = None):
    fps = 0
    total_frames = 0
    sum_time = 0
    count = 0

    if in_path is not None:
        cap = cv2.VideoCapture(in_path)
        #video = cv2.VideoCapture(out_path)
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            fps = cap.get(cv2.CV_CAP_PROP_FPS)
            total = int(cap.get(cv2.CV_CAP_PROP_FRAME_COUNT))
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total)
    else:
        cap = cv2.VideoCapture(0)

    start = time.time()
    first = True

    id = 0
    depth_old = None
    starttt = timeit.default_timer()

    while cap.isOpened():
        if in_path is not None:
            pbar.update(1)
        id = id + 1
        #for a in range(1000):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, resolution)

            if in_path is not None and first:
                first = False
                if horizontal:
                    size = (2 * frame.shape[1], frame.shape[0])
                else:
                    size = (frame.shape[1], 2 * frame.shape[0])
                result = cv2.VideoWriter(out_path,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, size)#size = (1440, 640)

            start = time.time()
            (right_img,depth_old,output) = generate_right(frame, model, depth_old,
                                                       theta, maxdeviation)
            #cv2.imshow('CV2Frame', frame)
            #cv2.imshow('CV2Frame2', right_img)

            if horizontal:
                vis = np.concatenate((frame, right_img), axis=1)
            else:
                vis = np.concatenate((frame, right_img), axis=0)

            if show_frames:
                cv2.imshow('vis',vis)

            if out_path is not None:
                result.write(vis)
            if show_depth:
                rescaled = 255.0 * (output - np.min(output)) / (np.max(output) - np.min(output))
                rescaled = rescaled.astype(np.uint8)
                cv2.imshow('depth', rescaled)

            if show_frames:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()

            end = time.time()
            sum_time = sum_time + (end - start)
            count = count + 1
            if show_fps and sum_time > 1:
                #print(end - start)
                print(count)
                sum_time = 0
                count = 0

        # Break the loop
        else:
            break

    if in_path is not None:
        pbar.close()
    stopttt = timeit.default_timer()
    print('Time: ', stopttt - starttt)

    cap.release()

    """
    if out_path is not None:
        result.release()
        if include_sound:
            sgeutils.add_sound(in_path, out_path, out_path_sound)
    """

