import numpy as np

import cv2

def detectCircles(im, radius, useGradient):
    '''
    Uses the Hough Transform to detect circles of a given radius.
    useGradient is a flag to optionally use edge gradient directions.
    
    Returns the (x, y) coordinates of detected circle centers.
    '''

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=radius-5, maxRadius=radius+5)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[:, :2]  # Return only (x, y) coordinates
    else:
        return np.array([])  # No circles detected
    


if __name__ == "__main__":
    input_image_path1 = "sports_balls.jpg"
    im = cv2.imread(input_image_path1)

    input_image_path2 = "eyes_deer.jpg"
    im_noisy = cv2.imread(input_image_path2)

    radius = 20

    circles = detectCircles(im, radius, useGradient=True)
    print(f"Detected circles in {input_image_path1}: {circles}")

    circles_noisy = detectCircles(im_noisy, radius, useGradient=True)
    print(f"Detected circles in {input_image_path2}: {circles_noisy}")

    # Visualize results
    for (x, y) in circles:
        cv2.circle(im, (x, y), radius, (0, 255, 0), 2)
    for (x, y) in circles_noisy:
        cv2.circle(im_noisy, (x, y), radius, (0, 255, 0), 2)
    cv2.imwrite("detected_circles_sports_balls.png", im)
    cv2.imwrite("detected_circles_eyes_deer.png", im_noisy)



