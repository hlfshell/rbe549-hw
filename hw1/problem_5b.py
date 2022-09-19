import cv2

img = cv2.imread('selfie.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 11)

_, binary_threshold = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY)

adaptive_threshold_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_threshold_gaussian = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite("imgs/5b_binary_threshold.jpg", binary_threshold)
cv2.imwrite("imgs/5b_adaptive_threshold_mean.jpg", adaptive_threshold_mean)
cv2.imwrite("imgs/5b_adaptive_threshold_gaussian.jpg", adaptive_threshold_gaussian)