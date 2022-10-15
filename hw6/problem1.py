import cv2
import numpy as np

MAX_FEATURES = 500
BEST_MATCH_PERCENTAGE = 0.05
FINAL_SIZE = (938, 1366)

# img = cv2.imread("./imgs/cove_full.png", cv2.IMREAD_COLOR)

# print("Showing the original image")
# cv2.imshow("Original", img)
# cv2.waitKey(0)

print("Showing each piece of the cut-up image")
# img_left = cv2.imread("./imgs/roshar_map_left.png", cv2.IMREAD_COLOR)
# img_middle = cv2.imread("./imgs/roshar_map_middle.png", cv2.IMREAD_COLOR)
# img_right = cv2.imread("./imgs/roshar_map_right.png", cv2.IMREAD_COLOR)
img_left = cv2.imread("./imgs/cove_left.png", cv2.IMREAD_COLOR)
img_middle = cv2.imread("./imgs/cove_middle.png", cv2.IMREAD_COLOR)
img_right = cv2.imread("./imgs/cove_right.png", cv2.IMREAD_COLOR)

print(img_left.shape, img_middle.shape, img_right.shape)

for index, img in enumerate([img_left, img_middle, img_right]):
    cv2.imshow(f"Image Segment {index +1}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

print("Using ORB to detect features")
orb = cv2.ORB_create(MAX_FEATURES)

# Now utilize ORB on each of the input image's segments
keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
keypoints_middle, descriptors_middle = orb.detectAndCompute(img_middle, None)
keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)

# Now find the matches between left/middle descriptors
# matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Brute force?
left_middle_matches = matcher.match(descriptors_left, descriptors_middle, None)
middle_right_matches = matcher.match(descriptors_middle, descriptors_right, None)

left_middle_matches = sorted(left_middle_matches, key=lambda match: match.distance, reverse=False)
middle_right_matches = sorted(middle_right_matches, key=lambda match: match.distance, reverse=False)

# Limit the matches to the top 15%
left_middle_matches = left_middle_matches[:int(BEST_MATCH_PERCENTAGE * len(left_middle_matches))]
middle_right_matches = middle_right_matches[:int(BEST_MATCH_PERCENTAGE * len(middle_right_matches))]

print("Displaying match results")
match_results = cv2.drawMatches(img_left, keypoints_left, img_middle, keypoints_middle, left_middle_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Match Results", match_results)
cv2.waitKey(0)

match_results = cv2.drawMatches(img_middle, keypoints_middle, img_right, keypoints_right, middle_right_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Match Results", match_results)
cv2.waitKey(0)

# Now we combine the images into a singular image.
print("Combining images")
points_left = np.zeros((len(left_middle_matches), 2))
points_middle = np.zeros((len(left_middle_matches), 2))

for index, match in enumerate(left_middle_matches):
    points_left[index,:] = keypoints_left[match.queryIdx].pt
    points_middle[index,:] = keypoints_middle[match.trainIdx].pt

homography, mask = cv2.findHomography(points_left, points_middle, cv2.RANSAC)
img_left_middle = cv2.warpPerspective(img_middle, np.linalg.inv(homography), FINAL_SIZE)

# Place the left image flat in the left side to overlap.
img_left_middle[0:img_left.shape[0],0:img_left.shape[1]] = img_left
cv2.imshow("Test", img_left_middle)
cv2.waitKey(0)

# Now we do the final right image. We'll be redoing out keypoints and matches based on
# our current warped image instead of using the base, as everything gets offset by
# the homography transformation
keypoints_left_middle, descriptors_left_middle = orb.detectAndCompute(img_left_middle, None)
keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)
left_middle_right_matches = matcher.match(descriptors_left_middle, descriptors_right, None)

# Limit the matches to the top 15%
left_middle_right_matches = sorted(left_middle_right_matches, key=lambda match: match.distance, reverse=False)
left_middle_right_matches = left_middle_right_matches[:int(0.10 * len(left_middle_right_matches))]

match_results = cv2.drawMatches(img_left_middle, keypoints_left_middle, img_right, keypoints_right, left_middle_right_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Match Results", match_results)
cv2.waitKey(0)

points_left_middle = np.zeros((len(left_middle_right_matches), 2))
points_right = np.zeros((len(left_middle_right_matches), 2))

for index, match in enumerate(left_middle_right_matches):
    points_left_middle[index,:] = keypoints_left_middle[match.queryIdx].pt
    points_right[index,:] = keypoints_right[match.trainIdx].pt

homography, mask = cv2.findHomography(points_left_middle, points_right, cv2.RANSAC)
final_img = cv2.warpPerspective(img_right, np.linalg.inv(homography), FINAL_SIZE)

# Create our mask
# _, mask = cv2.threshold(cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
# mask = cv2.bitwise_not(mask)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)
# final_img = cv2.bitwise_and(final_img, final_img, mask=mask)

# cv2.imshow("final chunk", mask)
# cv2.waitKey(0)
# dst = cv2.add(img1_bg,img2_fg)
# final_img = cv2.add(img_left_middle, final_img)
# final_img[0:img_left_middle.shape[0],0:img_left_middle.shape[1]] = img_left_middle
for c_index, c in enumerate(final_img):
    for r_index, pixel in enumerate(c):
        # print("pixel", pixel)
        # if pixel != [0,0,0]:
        if np.array_equal(pixel, [0,0,0]):
            final_img[c_index][r_index] = img_left_middle[c_index][r_index]

print(final_img.shape)

cv2.destroyAllWindows()
cv2.imshow("Final Image", final_img)
cv2.waitKey(0)