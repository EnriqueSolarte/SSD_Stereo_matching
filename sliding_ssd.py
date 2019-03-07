import numpy as np
from PIL import Image
import cv2
import os
from file_utilities import list_directories, save_obj
import numpy as np
import math
from tqdm import tqdm


def stereo_match(up_img, down_img, kernel, max_offset, baseline):
    # Load in both images, assumed to be RGBA 8bit per channel images
    up_img = np.asarray(up_img)
    # cv2.imshow("left", left)
    down_img = np.asarray(down_img)
    # cv2.imshow("right", right)
    # cv2.waitKey(0)
    h, w = up_img.shape  # assume that both images are same size

    # Depth (or disparity) map
    disp = np.zeros((w, h), np.uint8)
    disp.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
    tbar = tqdm(total=(w - kernel) * (h - kernel) * max_offset)
    for x in range(kernel_half, w - kernel_half):
        # print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for y in range(kernel_half, h - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for u in range(-kernel_half, kernel_half):
                    for v in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(up_img[y + v, x + u]) - int(down_img[y + v - offset, x + u])
                        ssd += ssd_temp * ssd_temp

                        # if this value is smaller than t6he previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                tbar.update(1)

            # set depth output for this x,y location to the best match
            disp[y, x] = best_offset
        save_obj("disp_1", disp)

    tbar.close()


if __name__ == '__main__':
    data_path = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
    dir_depth_map = os.path.join(data_path, "depth_up")
    dir_rgb_map_up = os.path.join(data_path, "image_up")
    dir_rgb_map_down = os.path.join(data_path, "image_down")

    list_depth_maps = list_directories(dir_depth_map)
    list_rgb_maps_up = list_directories(dir_rgb_map_up)
    list_rgb_maps_down = list_directories(dir_rgb_map_down)

    i = 0
    depth_map = np.load(os.path.join(dir_depth_map, list_depth_maps[i]))[:, :, 0]
    rgb_map_up = cv2.imread(os.path.join(dir_rgb_map_up, list_rgb_maps_up[i]))
    rgb_map_down = cv2.imread(os.path.join(dir_rgb_map_down, list_rgb_maps_down[i]))

    img_up = cv2.cvtColor(rgb_map_up, cv2.COLOR_BGR2GRAY)
    img_down = cv2.cvtColor(rgb_map_down, cv2.COLOR_BGR2GRAY)
    baseline = 0.2
    stereo_match(img_up, img_down, 6, 70, baseline)  # 6x6 local search kernel, 30 pixel search range
