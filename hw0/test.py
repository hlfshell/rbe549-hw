import cv2
import imutils

image = cv2.imread("test.jpg")
(h, w, d) = image.shape
print(f"width={w}, height={h}, depth={d}")

cv2.imshow("Image", image)
cv2.waitKey(0)