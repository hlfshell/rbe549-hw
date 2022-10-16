import cv2
import numpy as np
from matplotlib import pyplot as plt

# First load the image
img = cv2.imread("./imgs/dot-blot.jpg", cv2.IMREAD_COLOR)

# # We need the image in grayscale for sobel
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./imgs/gray.jpg", img)

# Then it's generally accepted that blurring helps edge
# detection techniques
blurred = cv2.GaussianBlur(img, (3,3), 0)

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
    blurred = cv2.GaussianBlur(img, (15,15), sigma)
    mh = cv2.Laplacian(blurred, cv2.CV_8UC1)
    _, threshold = cv2.threshold(mh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Show our result
    cv2.imshow(f"MH @ Sigma = {sigma}", threshold)
    cv2.waitKey(0)
    cv2.imwrite(f"./imgs/mh-{sigma}.jpg", threshold)

# Now let's do the fourier transform filter
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shifted[:,:,0], dft_shifted[:,:,1])))

cv2.imwrite("./imgs/dft.jpg", magnitude_spectrum)

rows, cols = img.shape
center = int(rows/2), int(cols/2)
r = 30
mask = np.ones((rows, cols, 2), np.uint8)
x, y = np.ogrid[:rows, :cols]

mask_area = (x- center[0]) ** 2 + (y-center[1]) ** 2 <= r*r
mask[mask_area] = 0

fshift = dft_shifted * mask
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.figure()
plt.imshow(fshift_mask_mag)
plt.show()
plt.close()

plt.figure()
plt.imshow(img_back)
plt.show()
plt.close()