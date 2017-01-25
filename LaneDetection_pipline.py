#importing some useful packages
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    ##############################  Filtering/Grouping multiple lines down to 2 lines that cover a lane FOV ###############
    # Group broken lines 
    group_lines, group_slopes, group_interxs, FoundBothLines = GroupLineLeftRight(lines)
    # TODO try adaptive/global thresholding if missing lane
    if not FoundBothLines:
        # some lane missing on this frame
        print("WARNING: Lane missing detected...")
    
    else:
        # normal detection 
        # Should only have 2
        mean_lines = RemoveOutlierMeanLines(group_lines, group_slopes, group_interxs)
        
        # Convert lines slope/intersection to 2 points for each line 
        # Note: reverse slope show on the image - y is now downward!
        # slope + is Right, - slope is Left
        # draw from the bottom to end of edge
        ImageShape = [img.shape[0],img.shape[1]]
        lines_xy = FindLineROI_XY(mean_lines, [330, img.shape[0]], img.shape)
        #lines_xy = FindLineROI_XY(mean_lines, [min_y, img.shape[0]], img.shape)
        lines_xy = np.array(lines_xy).reshape( (len(lines_xy), 1, 4) ) 
        lines = lines_xy
                
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=15)       
    return line_img, FoundBothLines

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

########################################## Lines Clean up ##########################################
def OtsuThresholding(image_blure):
    ret,binary = cv2.threshold(image_blure,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary


def ColorSelection(image, RGBthreshold):
    color_select = np.copy(image)
    red_threshold = RGBthreshold[0]
    green_threshold = RGBthreshold[1]
    blue_threshold = RGBthreshold[2]
    
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
                | (image[:,:,1] < rgb_threshold[1]) \
                | (image[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0,0,0]  
    return color_select


def GroupLineLeftRight(lines):
    """
        Sort out points to Left or Right line and compute corresponding a slope and intersection point
        It is also indicating if both lines are found or not
    """
    slopes = []
    intersecs = []
    FoundBoth = False
    for i in range(len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        # TODO save guard zero division from the vertical line
        try:
            slope = float(y2-y1)/float(x2-x1)
            
        except ZeroDivisionError:
            slope = float('Inf')
            
        slopes.append(slope)
        intersec = y2 - slope*x2
        intersecs.append(intersec)
        
    slopes = np.array(slopes)
    intersecs = np.array(intersecs)
    # labelling left right lines
    left_lines_index = [i for i, m in enumerate(slopes) if m < 0 and m != 'inf']
    right_lines_index = [i for i, m in enumerate(slopes) if m >= 0 and m != 'inf']
    # organising data
    group_lines  = np.array([np.array(left_lines_index), np.array(right_lines_index)])
    group_slopes = np.array([slopes[left_lines_index],slopes[right_lines_index]])
    group_inters = np.array([intersecs[left_lines_index],intersecs[right_lines_index]])
    
    # check if both left/right found
    if left_lines_index != [] and right_lines_index != []:
        FoundBoth = True
        
    return group_lines, group_slopes, group_inters, FoundBoth


def RemoveOutlierMeanLines(group_lines, group_slopes, group_inters, error_threshold = 0.3):
    """
        Clean up outlier points by checking slope
        Any error above/below thresholding from the mean will be thought of outliers
        This function remove outlier from slope only
    """
    mean_lines = []
    
    for data_line, data_slope, data_inter in zip(group_lines, group_slopes, group_inters):

        # remove outlier error > 2*std
        slope_std  = np.std(data_slope)
        slope_mean = np.mean(data_slope)
        slope_error = abs(data_slope - slope_mean)    
        ok_index = [slope_error <= error_threshold]     # anything beyond 0.3 are bad
        
        # mean slope and intersection without outlier
        mean_m = np.mean(data_slope[ok_index])
        mean_b = np.mean(data_inter[ok_index])
        mean_lines.append( (mean_m, mean_b) )
    return mean_lines

def FindLineROI_XY(mean_lines_params,Ymin_max, ImageShape):
    """
        Finding extreme bounding box points 
    """
    # slope < 0 is Left line, slope >=0 is Right line
    # inf slope is filtered out beforehand
    # TODO safe guard zero division just in case
    filter_lines = []
    for i in range(len(mean_lines_params)):
        slope = mean_lines_params[i][0]
        C = mean_lines_params[i][1]
        y1 = int(Ymin_max[0])
        y2 = int(Ymin_max[1])
        
        # check division
        try:
            x1 = int((y1-C)/slope)
        except ZeroDivisionError:
            print('inf slope abnormality')
            x1 = 0
            
        try:
            x2 = int((y2-C)/slope)
        except ZeroDivisionError:
            print('inf slope abnormality')
            x2 = 0
            
        #Check image bound
        if(x1 > ImageShape[1]): x1 = ImageShape[1]
        elif(x1 < 0): x1 = 0
        
        if(x2 > ImageShape[1]): x2 = ImageShape[1]
        elif(x2 < 0): x2 = 0
        
        if(y1 > ImageShape[0]): y1 = ImageShape[0]
        elif(y1 < 0): y1 = 0
        
        if(y2 > ImageShape[0]): y2 = ImageShape[0]
        elif(y2 < 0): y2 = 0
            
        filter_lines.append([x1,y1,x2,y2])
    return filter_lines


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    
    # RGB color selection
    RGBthreshold = [160,140,10]
    
    # blure iamge
    kernel_size = 11
    # edge canny
    low_threshold = 50
    high_threshold = 150
    # ROI- fixed don't change
    imshape = image.shape
    vertices = np.array([[
                          (120,imshape[0]),
                          (450, 320),
                          (525, 320),
                          (910,imshape[0])
                          ]],dtype=np.int32)
    
    # Hough Transform
    rho = 1                         # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180             # angular resolution in radians of the Hough grid
    threshold = 20                  # minimum number of votes (intersections in Hough grid cell) #25
    min_line_len = 10               # minimum number of pixels making up a line (#15)
    max_line_gap = 15               # maximum gap in pixels between connectable line segments #15
    
    # Lane detection processing
    extarcted_RGB =  ColorSelection(image, RGBthreshold)
    # Convert to gray scale
    gray = grayscale(extarcted_RGB)
    # Filtering noise Gaussian blur (13, 50, 150)
    blure_gray = gaussian_blur(gray, kernel_size)
    # Canny edge detection
    edges = canny(blure_gray, low_threshold, high_threshold)
    # select lane ROI
    masked_edges = region_of_interest(edges, vertices)

    # Finding lane lines- Hough Transform
    FoundBothLine = False
    line_image, FoundBothLine = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
     
    # Overlay lines
    color_edges = np.dstack((edges, edges, edges))
    line_edges = weighted_img(color_edges, line_image, alpha=0.8, beta=1., gamma=0.)
    line_rgb = weighted_img(image, line_image, alpha=0.8, beta=1., gamma=0.)
    return line_rgb


###############################   Main  ###################################
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
filenames = os.listdir("test_images/")
for image_file in filenames:
    # Read test images
    file_path = "test_images/" + image_file
    image = mpimg.imread(file_path)      
    
    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    
    # Process images for Lane detection
    line_rgb = process_image(image)
    
    # save to local drive
    output_name = image_file.split('.')[0]
    output_path = "test_images/" + output_name + "_out"
    mpimg.imsave(output_path, line_rgb, format='jpg')





