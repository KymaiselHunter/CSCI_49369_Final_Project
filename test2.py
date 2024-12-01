
import cv2
import numpy as np
from collections import Counter
import math
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
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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


def sobel_hough_fill_holes(image_path, threshold=127, output_path='binary_output_filled.png'):


    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) 


    sobel_x = cv2.Sobel(gray_thresh, cv2.CV_64F, 1, 0, ksize=3)  #Horizontal edges
    sobel_y = cv2.Sobel(gray_thresh, cv2.CV_64F, 0, 1, ksize=3)  #Vertical edges


    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude)

    #ret, gradient_thresh = cv2.threshold(gradient_magnitude, 150, 255, cv2.THRESH_BINARY) 

    line_image = np.zeros_like(gray)

    lines = cv2.HoughLines(gradient_magnitude, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=300)

    #get angles
    normal_angles = []    
    for line in lines:
        r, theta = line[0]
        normal_angles.append(np.degrees(theta) % 180)


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
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
        
    #print(normal_angles)
    #find dom angle
    angle_counter = Counter(normal_angles)
    dom_angle = max(angle_counter, key=angle_counter.get)
    print(dom_angle)
    
    roatated_original = rotation(image, dom_angle - 90)

    DISPLAY_WIDTH = 500
    #cv2.imshow('omg', image)
    cv2.imshow('gray', ResizeWithAspectRatio(image=gray, width=DISPLAY_WIDTH))
    cv2.imshow('gray_thresh', ResizeWithAspectRatio(image=gray_thresh, width=DISPLAY_WIDTH))
    cv2.imshow('Gradient Magnitude (Sobel)', ResizeWithAspectRatio(image=gradient_magnitude, width=DISPLAY_WIDTH))
    cv2.imshow('Hough', ResizeWithAspectRatio(image=line_image, width=DISPLAY_WIDTH))
    cv2.imshow('Rotate', ResizeWithAspectRatio(image=roatated_original, width=DISPLAY_WIDTH))

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    #cv2.imwrite(output_path, binary_filled)
    #print(f"Filled binary image saved as {output_path}")

if __name__ == "__main__":
    
    sobel_hough_fill_holes('./images/image.jpg', threshold=127, output_path='binary_output_filled.png')
