import cv2

video = cv2.VideoCapture(0)
  
while(True):
    _, frame = video.read()
    cv2.imshow('frame', frame)
    # cv2.imshow('Input', frame)
    if cv2.waitKey(0):
        break
  
video.release()
cv2.destroyAllWindows()