import cv2
import numpy as np
import urllib.request as urllib
#import imutils.video as VideoStream

cap = cv2.VideoCapture(1)
fg = cv2.createBackgroundSubtractorMOG2()

url='http://10.1.240.143:4747/mjpegfeed?640'

while True:
    _, frame = cap.read()
##    imgResp=urllib.urlopen(url)
##    print(imgResp)
##    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
##    frame=cv2.imdecode(imgNp,-1)
    fgmask = fg.apply(frame)
    temp_intensity_vals = []
    for i in range(frame.shape[0]):
    	for j in range(frame.shape[1]):
    		if(fgmask[i,j] not in temp_intensity_vals):
    			temp_intensity_vals.append(fgmask[i,j])
    print(temp_intensity_vals)

    cv2.imshow("Origi",frame)
    cv2.imshow("Fg",fgmask)

    k = cv2.waitKey(30) & 0xff
    if(k == 27):
        break

cap.release()
cv2.destroyAllWindows()
