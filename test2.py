
import cv2
import numpy as np
from collections import Counter
import math
import OutputKeyboard as key
#import Counter fromcollections #for coun ter

# nice resizeing 
# stolen from : https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


#nice rotaiton
# stolen from : https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

# given  rho and theta, we draw a line on the image
def drawLineFromHough(image, r, theta):
    # Stores the value of cos(theta) in a
      a = np.cos(theta)

      # Stores the value of sin(theta) in b
      b = np.sin(theta)

      # x0 stores the value rcos(theta)
      x0 = a*r

      # y0 stores the value rsin(theta)
      y0 = b*r

      # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
      x1 = int(x0 + 10000*(-b))

      # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
      y1 = int(y0 + 10000*(a))

      # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
      x2 = int(x0 - 10000*(-b))

      # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
      y2 = int(y0 - 10000*(a))

      # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
      # (0,0,255) denotes the colour of the line to be
      # drawn. In this case, it is red.
      cv2.line(image, (x1, y1), (x2, y2), 255, 2)

# morphlogical processed Image to extend edges of the piano togeter
def getMorphologicalProcessedImage(image):
    # Define an elongated kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Horizontal extension
    dilated = cv2.dilate(image, kernel, iterations=1)

    # Define an elongated kernel for closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Adjust kernel size
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    _, processed = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)

    return processed


#https://stackoverflow.com/questions/56589691/how-to-leave-only-the-largest-blob-in-an-image
# gets the biggest object in the image, this object should be the piano keys
def getBiggestBlob(image):
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(image.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(image, out)

# returns the bounds of the piano so we can only look at the keys
# assume the piano is already correctly rotated
def getCropBounds(image):
    # Find rows that contain at least one white pixel (255)
    rows_with_white = np.where(np.any(image == 255, axis=1))[0]

    bottom = rows_with_white[-1]
    top = rows_with_white[0]

    cols_with_white = np.where(np.any(image == 255, axis=0))[0]

    left = cols_with_white[0]
    right = cols_with_white[-1]

    return bottom, top, left, right


# crops the image to given parameter bounds
# this will allow us to work with only the piano keys in the image
def cropImage(image, top_row, bottom_row, left_col, right_col):
    # Crop the image using numpy slicing
    cropped_image = image[top_row:bottom_row, left_col:right_col]
    return cropped_image

#edited code from https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
def black_keys(cropped):
    #make the black keys highlighted
    thres_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thres_cropped = cv2.blur(thres_cropped,(3,3))
    _, new_cropped = cv2.threshold(thres_cropped,127,255,cv2.THRESH_BINARY_INV)
    
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    opening = cv2.morphologyEx(new_cropped,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow('thres_cropped', opening)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)


    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1


    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    temp = cropped

    markers = cv2.watershed(temp,markers)
    temp[markers == -1] = [255,255,0]

    highlighted_image = cropped.copy()
    black_key_coords = []

    #for loop to traverse through the "object" area
    for marker in range(2, markers.max() + 1):
        #create a mask for each segment
        mask = (markers == marker).astype(np.uint8) * 255

        #find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        
        for contour in contours:
            #get the "edge" of the object
            x, y, w, h = cv2.boundingRect(contour)

            #store the coordinates
            black_key_coords.append((x, y, w, h))  

            #draw rectangle on the highlighted image
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('cropped_black_keys', temp)
    cv2.imshow('testy', highlighted_image)

    return black_key_coords
    
    
    
    




def sobel_hough_fill_holes(image_path, threshold=127, output_path='binary_output_filled.png'):

    # read image 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # gray scale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #threshold the gray to get a binary image, we only want the white piano keys really
    ret, gray_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) 

    # perform a quick edge detection
    sobel_x = cv2.Sobel(gray_thresh, cv2.CV_64F, 1, 0, ksize=3)  #Horizontal edges
    sobel_y = cv2.Sobel(gray_thresh, cv2.CV_64F, 0, 1, ksize=3)  #Vertical edges


    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))


    # perform the hough transformation, where we will use this to find the dominant orientation
    line_image = np.zeros_like(gray)

    lines = cv2.HoughLines(gradient_magnitude, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=200)

    #get all angles
    normal_angles = []    
    for line in lines:
        rho, theta = line[0]
        normal_angles.append(np.degrees(theta) % 180)

        drawLineFromHough(line_image,rho,theta=theta)

        
    #calculate the dominant angle

    #print(normal_angles)
    #find dom angle
    #angle_counter = Counter(normal_angles)
    #dom_angle = max(angle_counter, key=angle_counter.get)

    angle_bins = np.histogram(normal_angles, bins=18, range=(0, 180))
    dom_angle = angle_bins[1][np.argmax(angle_bins[0])]

    print(dom_angle)
    
    # display all of the images so far
    DISPLAY_WIDTH = 500
    #cv2.imshow('omg', image)
    cv2.imshow('gray', ResizeWithAspectRatio(image=gray, width=DISPLAY_WIDTH))
    cv2.imshow('gray_thresh', ResizeWithAspectRatio(image=gray_thresh, width=DISPLAY_WIDTH))
    cv2.imshow('Gradient Magnitude (Sobel)', ResizeWithAspectRatio(image=gradient_magnitude, width=DISPLAY_WIDTH))
    cv2.imshow('Hough', ResizeWithAspectRatio(image=line_image, width=DISPLAY_WIDTH))


    # perform the roation in respect to the dominant angle, and display the results

    rotated_original = rotation(image, dom_angle - 90)
    rotated_gradiant = rotation(gradient_magnitude, dom_angle - 90)


    cv2.imshow('Rotate OG', ResizeWithAspectRatio(image=rotated_original, width=DISPLAY_WIDTH))
    cv2.imshow('Rotate Grad', ResizeWithAspectRatio(image=rotated_gradiant, width=DISPLAY_WIDTH))

    # use morphlogical image processing to extend edge lines horizontally 
    # which is then blobbed together, allowing us to find the piano keys, as they will be the
    # biggest blob
    # needed the internet to figure this part out
    ## defining the kernel i.e. Structuring element 

    processed = getMorphologicalProcessedImage(rotated_gradiant)
    cv2.imshow('Closing', ResizeWithAspectRatio(processed))

    blob = getBiggestBlob(processed)
    cv2.imshow('blob', blob)


    # once the keys are found, we can find the bounds of the keys and crop the roatated image
    bot, top,left,right = getCropBounds(blob)

    cv2.imshow('cropped', cropImage(rotated_original, top, bot, left, right))
   





    #Detecting black keys (testing different sizes of the cropped image)
    cropped = cropImage(rotated_original, top, bot, left, right)
    #bigger_cropped = cropImage(rotated_original, top-10, bot+10, left-10, right+10)
    smaller_cropped = cropImage(rotated_original, top+10, bot-10, left, right)
    #cv2.imshow('bigger_cropped', bigger_cropped)

    black_keys_val = black_keys(smaller_cropped)
    cv2.imshow('cropped_black_keys', cropped)
    #cv2.imshow('bigger_cr_black_keys', bigger_cropped)
    #cv2.imshow('smaller_cr_black_keys', black_keys(smaller_cropped))
    #cv2.imshow('test', cropped)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    #cv2.imwrite(output_path, binary_filled)
    #print(f"Filled binary image saved as {output_path}")


    




if __name__ == "__main__":
    
    sobel_hough_fill_holes('./images/image2.png', threshold=127, output_path='binary_output_filled.png')
    test = key.virtualKeyboard()
    test.start()
    test.add_note("C#", is_green=True)
    test.add_note("D#", is_green=True)
    test.add_note("F#", is_green=True)
    test.add_note("G#", is_green=True)
    test.add_note("A#", is_green=True)
    

