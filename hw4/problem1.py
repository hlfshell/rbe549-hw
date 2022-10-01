import cv2
import numpy as np
from math import floor, cos, sin

# First load the image
img = cv2.imread("./imgs/original.jpg", cv2.IMREAD_COLOR)

# Set our image size to a reasonable size
size = (500, floor(img.shape[1]/(img.shape[0]/500)))
# Note that the shape and size params are opposite!
# Hooray academic coding standards :-/
img = cv2.resize(img, (size[1], size[0]))

# We need the image in grayscale for sobel
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./imgs/gray.jpg", img)

# Then it's generally accepted that blurring helps edge
# detection techniques
blurred = cv2.GaussianBlur(img, (9,9), 0)

# Now let's generate the sobel edge detection
sobel = cv2.Sobel(blurred, cv2.CV_8UC1, dx=1, dy=1, ksize=5)
cv2.imwrite("./imgs/sobel.jpg", sobel)

# Now let's show our results
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("Sobel", sobel)
cv2.waitKey(0)

# Create a target image to draw on
target = np.copy(img)
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
cv2.imshow("target", target)
cv2.waitKey(0)

# Now that we we have the Sobel, let's find all the Hough Lines
# For theta we are using 1 degree, but it needs it in radians
lines = cv2.HoughLinesP(sobel, 1, np.pi / 180, 50, None, 200, 10)

if lines is not None:
    for line in lines:
        x0, y0, x1, y1 = line[0]
        start = (x0, y0)
        end = (x1, y1)
        cv2.line(target, start, end, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imwrite("./imgs/hough_lines_sobel.jpg", target)
cv2.imshow("Sobel Hough Lines", target)
cv2.waitKey(0)

# Then we use canny over sobel to improve the Hough transform results
canny = cv2.Canny(blurred, 10, 50)
cv2.imwrite("./imgs/canny.jpg", canny)
cv2.imshow("Canny", canny)
cv2.waitKey(0)

# Hough lines again, but using Canny
target = np.copy(img)
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 20, 15)

if lines is not None:
    for line in lines:
        x0, y0, x1, y1 = line[0]
        start = (x0, y0)
        end = (x1, y1)
        cv2.line(target, start, end, (0, 0, 255), 3, cv2.LINE_AA)

# Show the results again
cv2.imwrite("./imgs/hough_lines_canny.jpg", target)
cv2.imshow("Canny Hough Lines", target)
cv2.waitKey(0)

# Show the strongest line
target = np.copy(img)
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
x0, y0, x1, y1 = lines[0][0]
start = (x0, y0)
end = (x1, y1)
cv2.line(target, start, end, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imwrite("./imgs/hough_lines_peak_line.jpg", target)
cv2.imshow("Peak Line", target)
cv2.waitKey(0)