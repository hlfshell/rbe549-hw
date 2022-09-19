from random import gauss
import cv2

img = cv2.imread("./imgs/artwork.jpg", cv2.IMREAD_COLOR)

# Load and show the original image
cv2.imshow("Original Image", img)
cv2.waitKey(0)

# Box Filter with 𝑊 = 3 in both directions
box_filter_img = cv2.boxFilter(img, -1, (3,3))
cv2.imshow("Box Filter", box_filter_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/box_filtered_artwork.jpg", box_filter_img)

# Gaussian with σ=5
gaussian_img = cv2.GaussianBlur(img, (5, 5), 5)
cv2.imshow("Gaussian", gaussian_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/gaussian_artwork.jpg", gaussian_img)

# Median filter using a 5×5 window
median_img = cv2.medianBlur(img, 5)
cv2.imshow("Median", median_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/median_artwork.jpg", median_img)

# Box Filter with 𝑊 = 9 in both directions
box_filter_img = cv2.boxFilter(img, -1, (9,9))
cv2.imshow("Box Filter 3x", box_filter_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/3x_box_filtered_artwork.jpg", box_filter_img)

# Gaussian with σ=15
gaussian_img = cv2.GaussianBlur(img, (15, 15), 15)
cv2.imshow("Gaussian 3x", gaussian_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/3x_gaussian_artwork.jpg", gaussian_img)

# Median filter using a 15×15 window
median_img = cv2.medianBlur(img, 15)
cv2.imshow("Median 3x", median_img)
cv2.waitKey(0)
cv2.imwrite("./imgs/3x_median_artwork.jpg", median_img)