import cv2
import numpy as np
import urllib.request as urllib

##cap = cv2.VideoCapture(0)
fg = cv2.createBackgroundSubtractorMOG2()

url='http://10.1.240.143:8080/shot.jpg'

while True:
##    _, frame = cap.read()
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)
    fgmask = fg.apply(frame)

    cv2.imshow("Origi",frame)
    cv2.imshow("Fg",fgmask)

    k = cv2.waitKey(30) & 0xff
    if(k == 27):
        break

##cap.release()
cv2.destroyAllWindows()