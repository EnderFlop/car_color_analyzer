import cv2
import fpstimer

#init camera and set resolution
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 1920)
cam.set(4, 1080)

car_cascade = cv2.CascadeClassifier("cars.xml") #Thank you afzal442 on Github

def rotate_image(image, degrees):
  h, w = image.shape[:2]
  rotate_matrix = cv2.getRotationMatrix2D((w/2, h/2), degrees, 1)
  rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h))
  return rotated_image

timer = fpstimer.FPSTimer(30)
while True:
  ret, image = cam.read() #image is 480 x 640

  #rotate and crop image to get selection
  rotated_image = rotate_image(image, 50)
  cropped_image = rotated_image[400:1080, 830:1250] #constants defined by ME, where my webcam is.
  derotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

  #find and mark cars
  gray = cv2.cvtColor(derotated_image, cv2.COLOR_BGR2GRAY)
  cars = car_cascade.detectMultiScale(gray, 1.1, 1)
  for (x, y, w, h) in cars:
    cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 0, 255), 2)
  
  cv2.imshow("image1", gray)
  
  if cv2.waitKey(33) == 27: #wait for ESC
    break

  timer.sleep()

cv2.destroyAllWindows()