
import cv2
import numpy as np

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



    lines = cv2.HoughLinesP(gradient_magnitude, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=100, 
                            minLineLength=100, 
                            maxLineGap=10)

    
    line_image = np.zeros_like(gray)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)  # Draw white lines
    

    
    #cv2.imshow('omg', image)
    cv2.imshow('gray', gray)
    cv2.imshow('gray_threst', gray_thresh)
    cv2.imshow('Gradient Magnitude (Sobel)', gradient_magnitude)

    #cv2.imshow('sobel threst', gradient_thresh)

    cv2.imshow('Hough', line_image)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    #cv2.imwrite(output_path, binary_filled)
    #print(f"Filled binary image saved as {output_path}")

if __name__ == "__main__":
    
    sobel_hough_fill_holes('./images/image2.png', threshold=127, output_path='binary_output_filled.png')
