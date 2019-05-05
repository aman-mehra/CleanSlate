import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
from imutils.video import VideoStream
import time
# capture frames from a camera 
cap = VideoStream(src=1).start()
  
time.sleep(2.0)
# loop runs if capturing has been initialized 
while(1): 

	# im = cv2.imread("plank.jpg")
	im = cap.read()
	print(im)
	# im = im[1]

	# im = cv2.imread("plank.jpg")
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	_, bin = cv2.threshold(gray,120,255,1) # inverted threshold (light obj on dark bg)
	bin = cv2.dilate(bin, None)  # fill some holes
	bin = cv2.dilate(bin, None)
	bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
	bin = cv2.erode(bin, None)
	print(bin)
	contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# print(contours)
	rc = cv2.minAreaRect(contours[0])
	box = cv2.boxPoints(rc)
	for p in box:
	    pt = (p[0],p[1])
	    print (pt)
	    cv2.circle(im,pt,5,(200,0,0),2)
	cv2.imshow("plank", im)
	
	k = cv2.waitKey(5) & 0xFF
	if k == 27: 
		break


# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 