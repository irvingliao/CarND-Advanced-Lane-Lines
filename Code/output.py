#%% [markdown]
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---

#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import re
from Lane import Line, Lane
from collections import deque
from IPython import get_ipython

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

# Warp image by using Perspective Transform
def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Find the four below boundaries of the window
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
        (win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), 
        (win_xright_high, win_y_high), (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #: If you found > minpix pixels, recenter next window 
        # (`right` or `leftx_current`) on their mean position 
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def find_fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    #Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]).astype(int)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return out_img, left_fit, right_fit, left_fitx, right_fitx

# After we got polynomial fit values from the previous frame
def fit_poly(img_shape, leftx, lefty, rightx, righty):
     # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
    # Hint: consider the window areas for the similarly named variables in the previous quiz, but change the windows to our new search area
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    return left_fit, right_fit, left_fitx, right_fitx

#%% [markdown]
# ## Draw the Lane Line, calculate the curvature.
# 
#%%
prev_lanes = deque(maxlen=5)
dist_pickle = pickle.load( open( "../camera_cal/dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Calculates the curvature of polynomial functions in pixels.
def measure_curvature_pixels(ploty, left_fit, right_fit, xm_per_pix = (3.7/700)):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def polynomial_x(y, fit):
    return fit[0]*(y)**2 + fit[1]*y + fit[2]

def addText(img, texts, pos):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    current_y = pos[1]

    for text in texts:
        text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    
        x = pos[0]
        y = current_y + text_height
        current_y = y + 5
        bottomLeftCornerOfText = (x, y)

        cv2.putText(img, text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

def drawLine(warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp

def processImage(img):
    # Correct Distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Color Gradient threshold
    color_binary, cg_combined = processColorGradient(undist, s_thresh=(170, 250), h_thresh=(15, 100), sx_thresh=(20, 100), sy_thresh=(0, 255), mag_thresh=(30, 100), dir_thresh=(np.pi*30/180, np.pi*75/180), kernel=3)
    # mpimg.imsave('../output_images/test1_gradient.jpg', cg_combined)

    img_size = (img.shape[1], img.shape[0])
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
    
    # Get Warped Image by Perspective Transform
    warped = warper(cg_combined, src, dst)
    # mpimg.imsave('../output_images/test4_warp.jpg', warped)

    if len(prev_lanes) > 0:
        # If have previous Line Fit Value
        lastLane = prev_lanes[-1]
        left_fit, right_fit, left_fitx, right_fitx = search_around_poly(warped, lastLane.left_line.best_fit, lastLane.right_line.best_fit)

    else:
        # Find Polynomial Line Fit Value by Sliding Windows
        out_img, left_fit, right_fit, left_fitx, right_fitx = find_fit_polynomial(warped)

    ploty = np.linspace(0, img_size[1]-1, img_size[1])

    newLane = Lane()
    newLane.left_line.current_fit = left_fit
    newLane.left_line.recent_xfitted = left_fitx
    newLane.right_line.current_fit = right_fit
    newLane.right_line.recent_xfitted = right_fitx
    
    n_left = 1
    n_right = 1
    left_fit_total = np.array([0,0,0], dtype='float')
    right_fit_total = np.array([0,0,0], dtype='float')
    left_fitx_total = np.zeros_like(left_fitx)
    right_fitx_total = np.zeros_like(right_fitx)

    for prev in prev_lanes:
        left_fit_total += prev.left_line.best_fit
        right_fit_total += prev.right_line.best_fit
        left_fitx_total += prev.left_line.bestx
        right_fitx_total += prev.right_line.bestx
        n_left += 1
        n_right += 1
    
    left_fit_total += left_fit
    left_fitx_total += left_fitx
    right_fit_total += right_fit
    right_fitx_total += right_fitx

    if len(prev_lanes) > 0:
        # diff between last and new fit
        lastLane = prev_lanes[-1]
        far_y = img_size[1] / 2 + 100
        newLane.left_line.diffs = left_fit - lastLane.left_line.best_fit
        newLane.right_line.diffs = right_fit - lastLane.right_line.best_fit
        left_diff_x = polynomial_x(far_y, newLane.left_line.diffs)
        right_diff_x = polynomial_x(far_y, newLane.right_line.diffs)
        # print("diff left:", left_diff_x, "right:", right_diff_x )

        if np.absolute(left_diff_x) > 80:
            left_fit_total -= left_fit
            left_fitx_total -= left_fitx
            n_left -= 1

        if np.absolute(right_diff_x) > 80:
            right_fit_total -= right_fit
            right_fitx_total -= right_fitx
            n_right -= 1
    
    avg_left_fit = left_fit_total/n_left
    avg_left_fitx = left_fitx_total/n_left
    newLane.left_line.best_fit = avg_left_fit
    newLane.left_line.bestx = avg_left_fitx

    avg_right_fit = right_fit_total/n_right
    avg_right_fitx = right_fitx_total/n_right
    newLane.right_line.best_fit = avg_right_fit
    newLane.right_line.bestx = avg_right_fitx

    left_x = polynomial_x(img_size[1]-1, left_fit)
    right_x = polynomial_x(img_size[1]-1, right_fit)
    mid_lane_x = round( (left_x + right_x)/2 )
    mid_car_x = round( img_size[0]/2 )
    m_per_p = 3.7/(right_x-left_x)

    newLane.left_line.line_base_pos = (mid_car_x-left_x)*m_per_p
    newLane.right_line.line_base_pos = (right_x-mid_car_x)*m_per_p

    # difference from center, > 0 is on right side, < 0 is left
    diff_center = (mid_car_x - mid_lane_x)*m_per_p
    direction = "left"
    if diff_center > 0:
        direction = "right"
        
    diff_center_text = f"Vehicle is {np.absolute(diff_center):.2f}m " + direction + " of center"
    # print("diff from center:", diff_center)

    left_curv, right_curv = measure_curvature_pixels(ploty, left_fit, right_fit, m_per_p)
    newLane.left_line.radius_of_curvature = left_curv
    newLane.right_line.radius_of_curvature = right_curv
    curv_left_text = f"Left Curvature: {left_curv:.2f}m"
    curv_right_text = f"Right Curvature: {right_curv:.2f}m"
    # print("left curv:", left_curv)
    # print("right curv:", right_curv)

    prev_lanes.append(newLane)

    # Draw the lane line
    color_warp = drawLine(warped, newLane.left_line.bestx, newLane.right_line.bestx, ploty)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    addText(result, [curv_left_text, curv_right_text, diff_center_text], (0, 5))
    return result

#%%
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output = '../output_videos/project_video.mp4'
# clip1 = VideoFileClip("../project_video.mp4").subclip(20,24)
clip1 = VideoFileClip("../project_video.mp4")
clip = clip1.fl_image(processImage) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')

#%%
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output = '../output_videos/challenge_video.mp4'
clip1 = VideoFileClip("../challenge_video.mp4").subclip(0,5)
# clip1 = VideoFileClip("../challenge_video.mp4")
clip = clip1.fl_image(processImage) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')