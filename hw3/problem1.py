import cv2
from math import floor

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
sobel = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imwrite("./imgs/sobel.jpg", sobel)

# Now let's show our results
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("Sobel", sobel)
cv2.waitKey(0)

# Now let's do Marr-Hildreth edges, which is the Laplace
# of the Gaussian
for sigma in [1, 2, 4, 8, 16]:
    blurred = cv2.GaussianBlur(img, (9,9), sigma)
    mh = cv2.Laplacian(img, cv2.CV_64F, (7,7), scale=1)
    mh.convertTo(mh, cv2.CV_16UC1)
    cv2.threshold(mh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Show our result
    cv2.imshow(f"MH @ Sigma = {sigma}", mh)
    cv2.waitKey(0)
    cv2.imwrite(f"./imgs/mh-{sigma}.jpg", mh)