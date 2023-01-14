import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 1920)
cam.set(4, 1080)

while True:
  ret, img = cam.read()
  cv2.imshow("test", img)

  if cv2.waitKey(33) == 27: #wait for ESC
    break