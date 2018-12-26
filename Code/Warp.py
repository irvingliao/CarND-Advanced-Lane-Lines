import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

# Warp image by using Perspective Transform
def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

    return warped

#%%
imgNames = glob.glob('../output_images/*_color-gradient.jpg')
pattern = re.compile('/output_images/(.*)_color-gradient.jpg')

for fname in imgNames:
    image = cv2.imread(fname)
    img_size = (image.shape[1], image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 50), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    warped = warper(image, src, dst)
    name = pattern.search(fname).group(1)
    path = '../output_images/' + name + '_persp.jpg' 
    cv2.imwrite(path, warped)