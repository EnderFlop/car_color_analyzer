import cv2
import numpy as np
import fpstimer

#credit to https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/ for the vehicle detection help.

#init camera and set resolution
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 1920)
cam.set(4, 1080)

def prepare_image(image):
  h, w = image.shape[:2]

  #rotate image right 50 degrees
  rotate_matrix = cv2.getRotationMatrix2D((w/2, h/2), 50, 1)
  rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h))

  #crop image over road section
  cropped_image = rotated_image[400:1080, 800:1250] #constants defined by ME, where my webcam is.

  #rotate back 90 degrees to be "flat"
  derotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

  return derotated_image

timer = fpstimer.FPSTimer(30)
last_ret, last_image = cam.read()
timer.sleep()
while True:
  ret, image = cam.read() #image is 480 x 640

  #rotate and crop image to get selection (already done to previous frame)
  prepped_last_image = prepare_image(last_image)
  prepped_image = prepare_image(image)
  
  #make the images grayscale
  grayA = cv2.cvtColor(prepped_last_image, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(prepped_image, cv2.COLOR_BGR2GRAY)

  #compute the difference between the images to find objects in motion
  diff_image = cv2.absdiff(grayB, grayA)
  ret, threshold = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

  #dilate the objects to make them more clear, "blobby"
  kernel = np.ones((9,9), np.uint8)
  dilated = cv2.dilate(threshold, kernel, iterations = 5)

  contours, heirarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  main_contour = [None, 0]
  for contour in contours:
    if cv2.contourArea(contour) > main_contour[1]:
      main_contour = [contour, cv2.contourArea(contour)]

  cv2.drawContours(prepped_image, main_contour[0], -1, (127, 200, 0), 2)

  cv2.imshow("image1", prepped_image)
  
  if cv2.waitKey(33) == 27: #wait for ESC
    break

  last_ret, last_image = ret, image
  timer.sleep()

cv2.destroyAllWindows()