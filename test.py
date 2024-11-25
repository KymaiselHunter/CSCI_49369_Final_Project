
import cv2
import numpy as np

def sobel_hough_fill_holes(image_path, threshold=127, output_path='binary_output_filled.png'):


    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  #Horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  #Vertical edges


    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude)


    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 
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

    #Combine sobel gradient and hough lines
    combined = cv2.bitwise_or(gradient_magnitude, line_image)


    _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    #fill in the holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    #fill in holes
    binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    
    

    
    #cv2.imshow('Original Image', image)
    cv2.imshow('Gradient Magnitude (Sobel)', gradient_magnitude)
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('Hough Lines', line_image)
    ##cv2.imshow('Combined Edges and Lines', combined)
    #cv2.imshow('Binary Image Before Filling', binary)
    #cv2.imshow('Binary Image After Filling', binary_filled)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    cv2.imwrite(output_path, binary_filled)
    print(f"Filled binary image saved as {output_path}")

if __name__ == "__main__":
    
    sobel_hough_fill_holes('image2.png', threshold=127, output_path='binary_output_filled.png')
