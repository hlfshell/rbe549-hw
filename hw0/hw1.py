from time import sleep
import cv2

# First we take our selfie
print("Taking selfie...")
camera = cv2.VideoCapture(0)

frame = None
while True:
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    # Take the selfie as soon as we hit any key
    if key != -1:
        break
camera.release()

# Show the selfie we took
print("Showing selfie...")
cv2.imshow("Selfie", frame)
cv2.waitKey(0)

# Now showing the video
print("Showing overly dramatic video...")
video = cv2.VideoCapture('video.mp4')
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) != -1:
        break
    sleep(1/32)

cv2.destroyAllWindows()