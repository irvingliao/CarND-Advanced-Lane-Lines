#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, kernel=3):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))

    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelX**2 + sobelY**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def colorBinary(hsv, low, high):
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    h_binary = np.zeros_like(h)
    h_binary[(h >= low[0]) & (h <= high[0])] = 1

    s_binary = np.zeros_like(s)
    s_binary[(s >= low[1]) & (s <= high[1])] = 1

    v_binary = np.zeros_like(v)
    v_binary[(v >= low[2]) & (v <= high[2])] = 1

    color_binary = np.zeros_like(h)
    color_binary[(h_binary == 1) & (s_binary == 1) & (v_binary == 1)] = 1
    return color_binary

def processColorGradient(img, s_thresh=(170, 255), h_thresh=(15, 100), sx_thresh=(20, 100), sy_thresh=(40, 100), mag_thresh=(40, 100), dir_thresh=(0, np.pi/2), kernel=3):
    img = np.copy(img)
    
    sxbinary = abs_sobel_thresh(img, orient='x', thresh_min=sx_thresh[0], thresh_max=sx_thresh[1], kernel=kernel)
    sybinary = abs_sobel_thresh(img, orient='y', thresh_min=sy_thresh[0], thresh_max=sy_thresh[1], kernel=kernel)
    mag_binary = mag_threshold(img, sobel_kernel=kernel, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=kernel, thresh=dir_thresh)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]
    
    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Identify Yellow and White color by HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow = colorBinary(hsv, (5, 100, 100), (75, 255, 255))
    white = colorBinary(hsv, (19, 0, 255-72), (255, 72, 255))

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), s_binary, h_binary, sxbinary, sybinary, mag_binary, dir_binary)) * 255
    
    # Combine the binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[ ((s_binary == 1) & ((h_binary == 1) | (yellow == 1) | (white == 1)) ) | ((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) ] = 1
    return color_binary, combined_binary

#%%
import re
# Color Transfrom, Gradient with threshold

images = glob.glob('../test_images/*_undist.jpg')
pattern = re.compile('/test_images/(.*)_undist.jpg')

for fname in images:
    image = mpimg.imread(fname)
    result, combined = processColorGradient(image, 
                                s_thresh=(170, 250), h_thresh=(15, 100), 
                                sx_thresh=(20, 100), sy_thresh=(0, 255), 
                                mag_thresh=(30, 100), dir_thresh=(np.pi*30/180, np.pi*75/180), 
                                kernel=3)
    name = pattern.search(fname).group(1)
    path = '../output_images/' + name + '_color-gradient.jpg' 
    mpimg.imsave(path, combined, cmap='gray')