import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

KSIZE = 9
GRADMIN = 20
GRADMAX = 100
MAGMIN = 30
MAGMAX = 100
DIRMIN = 0.7
DIRMAX = 1.3
COLORMIN = 170
COLORMAX = 255
OFFSET = 85

MARGIN = 80

w,h = 1280,720
x,y = 0.5*w, 0.8*h
bottom_left = [1100./1280*w,720./720*h]
bottom_right = [835./1280*w,547./720*h]
top_left = [200./1280*w,720./720*h]
top_right = [453./1280*w,547./720*h]

sq_top_left = [(w-x)/2.,h]
sq_top_right = [(w-x)/2.,0.82*h]
sq_bottom_left = [(w+x)/2.,h]
sq_bottom_right = [(w+x)/2.,0.82*h]

src = np.float32([top_left,
                  top_right,
                  bottom_right,
                  bottom_left])

dst = np.float32([sq_top_left,
                  sq_top_right,
                  sq_bottom_right,
                  sq_bottom_left])


##############################################
############# CALIBRATE CAMERA ###############
# prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# declare arrays to store object and image points
objpoints = [] # 3d points
imgpoints = [] # 2d points

# Make a list of chessboard images
#images = glob.glob("/Users/grantrosario/Desktop/advanced_lane_finding/camera_cal/calibration*.jpg")
images = glob.glob("camera_cal/calibration*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # if found, add object and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
##############################################
##############################################


##############################################
############ CREATE BINARY IMAGE #############
def createBinary(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = KSIZE)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = KSIZE)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # Magnitude
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    # Direction
    direction = np.arctan2(abs_sobely, abs_sobelx)

    # Threshold Direction
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= DIRMAX) & (direction <= DIRMAX)] = 1

    # Threshold Magnitude
    mag_binary = np.zeros_like(scaled_sobelxy)
    mag_binary[(scaled_sobelxy >= MAGMIN) & (scaled_sobelxy <= MAGMAX)] = 1

    # Threshold x gradient
    gradx_binary = np.zeros_like(scaled_sobelx)
    gradx_binary[(scaled_sobelx >= GRADMIN) & (scaled_sobelx <= GRADMAX)] = 1

    # Threshold y gradient
    grady_binary = np.zeros_like(scaled_sobely)
    grady_binary[(scaled_sobely >= GRADMIN) & (scaled_sobelx <= GRADMAX)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= COLORMIN) & (s_channel <= COLORMAX)] = 1

    color_binary = np.dstack(( np.zeros_like(gradx_binary), gradx_binary, s_binary))

    # Combine binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (gradx_binary == 1) & (grady_binary == 1)] = 1
    # plt.imshow(combined_binary, cmap='gray')
    # plt.show()
    return (combined_binary)
##############################################
##############################################


##############################################
########### PERSPECTIVE TRANSFORM ############
def transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # dst_pts = np.array([sq_bottom_left,sq_bottom_right,sq_top_right,sq_top_left], np.int32)
    # dst_pts = dst_pts.reshape((-1,1,2))
    # src_pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
    # src_pts = src_pts.reshape((-1,1,2))
    # copy = img.copy()
    # cv2.polylines(copy, [dst_pts],True,(0,0,0), thickness=2)
    # cv2.polylines(copy, [src_pts],True,(0,0,0), thickness=2)
    # plt.imshow(warped, cmap='gray')
    # plt.show()
    # cv2.imshow('img', warped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (warped, M, Minv)
##############################################
##############################################

##############################################
############### FINDING LINES ################
def laneHistogramDetection(warped):
    # histogram of bottom half of image
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis = 0)

    # left and right histogram peaks
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    # height of windows
    window_height = np.int(warped.shape[0]/nwindows)

    # Identify x and y positions of non-zero pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Positions to update for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = MARGIN

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # empty lists to recieve left and right pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Rectangle data for visualization
    rectangle_data = []

    for window in range(nwindows):
        # Identify window boundaries in x, y, right, and left
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN

        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox
        >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox
        >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append indices to lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window around mean positon
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate array of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit second order polynomial to each
    left_fit = None
    right_fit = None

    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return (left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data)


    ##############################################
    ########### IF ALREADY FOUND #################
def laneHistogramSkipWindows(warped, left_fit_prev, right_fit_prev):
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - MARGIN)) &
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + MARGIN)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - MARGIN)) &
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + MARGIN)))

    # Extract pixel locations
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit polynomial for these locations
    new_left_fit = None
    new_right_fit = None
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        new_left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        new_right_fit = np.polyfit(righty, rightx, 2)


    return (new_left_fit, new_right_fit, left_lane_inds, right_lane_inds)


##############################################
############# MEASURE CURVATURE ##############
def measureCurveAndDist(binary_img, left_fit, right_fit, left_lane_inds, right_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Choose max y-value, corresponding to bottom of image
    h = binary_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Distance from center is image x midpoint - mean of left_fit and right_fit intercepts
    if right_fit is not None and left_fit is not None:
        car_position = binary_img.shape[1]/2
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return (left_curverad, right_curverad, center_dist)

##############################################
##############################################

##############################################
############### DRAW LINES ###################
def drawLines(img, warped, left_fit, right_fit, Minv):
    new_img = np.copy(img)
    if left_fit is None or right_fit is None:
        return img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = warped.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

##############################################
################ DRAW DATA ###################
def drawData(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img
##############################################
##############################################

##############################################
################# PIPELINE ###################
def pipeline(img):
    binary_img = createBinary(img)
    warped_img, M, Minv = transform(binary_img, src, dst)
    return (warped_img, Minv)

##############################################
##############################################

class Line():
    def __init__(self):
        # check if line was detected
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add found fit
        if fit is not None:
            if self.best_fit is not None:
                # compare new fit to current best fit
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # check fit queue for better fits, otherwise accept this fit
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

##############################################
##############################################

def process_image(base_img):
    new_img = np.copy(base_img)
    binary_img, Minv = pipeline(new_img)

    if not l_line.detected or not r_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds, _ = laneHistogramDetection(binary_img)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = laneHistogramSkipWindows(binary_img, l_line.best_fit,
                                                                                                    r_line.best_fit)
    if left_fit is not None and right_fit is not None:
        h = base_img.shape[0]
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        x_int_diff = abs(r_fit_x_int - l_fit_x_int)
        if abs(550 - x_int_diff) > 100:
            left_fit = None
            right_fit = None

    l_line.add_fit(left_fit, left_lane_inds)
    r_line.add_fit(right_fit, right_lane_inds)

    if l_line.best_fit is not None and r_line.best_fit is not None:
        img_out1 = drawLines(new_img, binary_img, l_line.best_fit, r_line.best_fit, Minv)
        rad_l, rad_r, d_center = measureCurveAndDist(binary_img, l_line.best_fit, r_line.best_fit,
                                                                left_lane_inds, right_lane_inds)
        img_out = drawData(img_out1, (rad_l+rad_r)/2, d_center)
    else:
        img_out = new_img

    return img_out


l_line = Line()
r_line = Line()
vid_output = 'first_attempt.mp4'
clip1 = VideoFileClip("project_video.mp4")
first_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
first_clip.write_videofile(vid_output, audio=False)

#img = cv2.imread("test_images/test5.jpg")
