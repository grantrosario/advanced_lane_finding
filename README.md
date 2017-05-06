
#**Advanced Lane Finding Project**
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/calibration_img.png "Undistorted"
[image2]: output_images/undistort_street.png "Road Transformed"
[image3]: output_images/binary.png "Road Binary"
[image4]: output_images/warped.jpg "Warped Lanes"
[image5]: output_images/warped_binary.png "Warped Binary Lanes"
[image6]: output_images/histogram.png "Histogram"
[image7]: output_images/find_lanes.png "Polynomial Lanes"
[image8]: output_images/find_lanes_margin.png "Margin Lanes"
[image9]: output_images/final.png "Output Image"
[video1]: final_attempt.mp4 "Video"


### Camera Calibration

#### 1. Description

The code for this step is contained in the second code cell of the IPython notebook located in "advanced\_lane\_finding.ipynb" (or in lines 44 through 71 of the file called `alf.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Undistort

The first step in the pipeline is to undistort the test image as seen below: 

![alt text][image2]

#### 2. Create thresholded binary image

I decided to avoid using gradients since they do not handle shadows well. Instead, I used a combination of color thresholds to generate a binary image (thresholding steps at lines 73 through 145 in `alf.py`). Specifically, I utilized the rgb color set to focus on yellow lines and the L channel in the HLS color set to focus on white lines. Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `transform()`, which appears in lines 150 through 159 in the file `alf.py` (or in the 4th code cell of the IPython notebook).  The `transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
w,h = 1280,720
x,y = 0.5*w, 0.8*h

src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
                  
dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720        | 
| 453, 547      | 320, 590.40002441      |
| 835, 547     | 960, 590.40002441      |
| 1100, 720      | 960, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. I then converted it to binary as well.

![alt text][image4]
![alt text][image5]

#### 4. Lane-line Identification

This step can be seen in lines 152 through 320 of the code in `alf.py`. I calculated the following histogram for the image which shows active pixels along all columns on the lower half of the image.

![alt text][image6] 

The two tallest peaks on the left half and right half show where in the image the lane lines appear. 
 
  
Next, I created 9 individual windows starting from the bottom and generating one by one and scanning their area for pixels. Using this, I was able to generate a fit for the left and right lane lines as seen below.

![alt text][image7] 

Once I was able to estimate this from scratch, I could use this estimation for future frames of the video instead of having to estimate the lines from scratch each time. I instead scanned around future areas using a margin of 100 starting from the point of the previous frame line. Below is a visualization of this margin mathod. 

![alt text][image8]

#### 5. Radius of Curvature and Distance Calculation

I did this in lines 323 through 371 in my code in `alf.py`

The radius of curvature was calculated based on the previously calculated second order polynomial.

I first converted the pixels to meters in accordance with U.S. regulations that lane width be 3.7 meters and our images have a lane _length_ of about 30 meters.

Once I do this I just fit the left and right lane to a polynomial and use the following formula to calculate the radius. 
 
```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
In order to calculate the distance from the center I just calculated the car position as the center of the image, subtracted the mean of the left and right intercepts from it and mutiplied that by the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 373 through 419 in my code in `alf.py` in the functions `drawLines()` and `drawData`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](final_attempt.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The biggest factors of this project were thresholding, skipping the sliding windows functionality if need be, and smoothing the fits. These were the 3 areas that took me the most time, had the largest effect, and I believe could be greatly improved upon.  
  
As stated, I used the sliding window functionality in order to read the lane lines and skipped this funcationaly when I could see the line based on the previous frame. I would like to see how using convolution would compare to this for possible improvement.

I also only used the RGB and L-channel of HLS for thresholding. This is the biggest weakness as the program does not fair well in shadows and noisy environemnts. I would definitely like to explore this in more detail and build a more robust threshold. One possibility would be to create a dynamic threshold by scanning the brightness of the image.
  
Lastly, smoothing the fits took a lot of trial and error. I ended up using an `add_fit()` method in the Line() class to average the fits and find the best one. I think this may be able to be improved upon by avoiding the hard-coded values I used.
