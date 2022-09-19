import cv2
from math import floor

# First load the image
img = cv2.imread("./imgs/original.jpg", cv2.IMREAD_COLOR)

# Set our image size to a reasonable size
size = (500, floor(img.shape[1]/(img.shape[0]/500)))
# Note that the shape and size params are opposite!
# Hooray academic coding standards :-/
img = cv2.resize(img, (size[1], size[0]))
cv2.imwrite("./imgs/gray.jpg", img)

# We need the image in grayscale for sobel
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Then it's generally accepted that blurring helps edge
# detection techniques
img = cv2.GaussianBlur(img, (9,9), 0)

# Now let's generate the sobel edge detection
sobel = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imwrite("./imgs/sobel.jpg", sobel)

# Now let's show our results
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("Sobel", sobel)
cv2.waitKey(0)

