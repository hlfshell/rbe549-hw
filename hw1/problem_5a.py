import cv2

from matplotlib import pyplot as plt

# Read in the image and get it to grayscale
img = cv2.imread('selfie.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create histogram of pixel counts
histograms = cv2.calcHist([img], [0], None, [256], [0, 256])
histogram = histograms[0]

# Now we isolate and figure out which bin we hit 80% of our
# pixels being set to white
height, width = img.shape
total_pixels = height * width
white_percentage = 0.8

white_pixels = 0
threshold_at = 0
for index, bin in enumerate(histograms):
    white_pixels += bin
    if white_pixels >= (1-white_percentage) * total_pixels:
        threshold_at = index
        break

print(f"Thresholding at any value below {threshold_at}")
binary_threshold = cv2.copyTo(img, None)
binary_threshold[binary_threshold > threshold_at] = 255
binary_threshold[binary_threshold != 255] = 0

cv2.imwrite("imgs/binary_threshold.jpg", binary_threshold)