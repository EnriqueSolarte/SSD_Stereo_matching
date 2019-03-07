import numpy as np
import math
from file_utilities import load_obj
import cv2

while True:
    disp = load_obj("disp_1")
    baseline = 0.2
    angle = np.zeros((512, 1024))
    angle2 = np.zeros((512, 1024))
    for i in range(1024):
        for j in range(512):
            theta_T = math.pi - ((j + 0.5) * math.pi / 512)
            angle[j, i] = baseline * math.sin(theta_T)
            angle2[j, i] = baseline * math.cos(theta_T)

    mask = disp > 0

    depth = np.zeros_like(disp).astype(np.float)
    depth[mask] = (angle[mask] / np.tan(disp[mask] / 180 * math.pi)) + angle2[mask]

    np.save("depth_est", depth)
    cv2.imshow("depth", depth)
    cv2.waitKey(0)
    break
