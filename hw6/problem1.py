import cv2
import numpy as np

MAX_FEATURES = 500
BEST_MATCH_PERCENTAGE = 0.15

img = cv2.imread("./imgs/roshar_map_full.jpg", cv2.IMREAD_COLOR)

print("Showing the original image")
cv2.imshow("Original", img)
cv2.waitKey(0)

print("Showing each piece of the cut-up image")
# img_left = cv2.imread("./imgs/roshar_map_left.png", cv2.IMREAD_COLOR)
# img_middle = cv2.imread("./imgs/roshar_map_middle.png", cv2.IMREAD_COLOR)
# img_right = cv2.imread("./imgs/roshar_map_right.png", cv2.IMREAD_COLOR)
img_left = cv2.imread("./imgs/cove_left.png", cv2.IMREAD_COLOR)
img_middle = cv2.imread("./imgs/cove_middle.png", cv2.IMREAD_COLOR)
img_right = cv2.imread("./imgs/cove_right.png", cv2.IMREAD_COLOR)

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
print("scores", left_middle_matches[0].distance, left_middle_matches[-1].distance)
middle_right_matches = sorted(middle_right_matches, key=lambda match: match.distance, reverse=False)
print("scores", middle_right_matches[0].distance, middle_right_matches[-1].distance)

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