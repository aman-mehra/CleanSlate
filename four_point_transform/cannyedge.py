import cv2, numpy as np
import sys
import time
from imutils.video import VideoStream
import writer
# capture frames from a camera 
cap = VideoStream(src=1).start()
  
time.sleep(2.0)

def get_TransformMatrix(transform_coordinates):
    rows,cols = 480,640
    pts1 = np.float32(transform_coordinates)
    pts2 = np.float32([[640,0],[640,640],[0,640]])

    M = cv2.getAffineTransform(pts1,pts2)
    return M

def sort_points(transform_coordinates):
    temp_arr = []
    for i in range(4):
        idx,minval = 0,transform_coordinates[0][0]
        for j in range(len(transform_coordinates)):
            if transform_coordinates[j][0] < minval:
                minval = transform_coordinates[j][0]
                idx = j
        temp_arr.append(transform_coordinates[idx])
        del transform_coordinates[idx]

    return temp_arr

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new,new)
    return new

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    transform_coordinates = []
    while True:
        orig = cap.read()
        ##    orig = cv2.imread(sys.argv[1])

        # these constants are carefully picked
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(img, (3,3), 0, img)


        # this is to recognize white on white
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.dilate(img, kernel)

        edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
        for line in lines[0]:
             cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                             (255,0,0), 2, 8)

        # finding contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_KCOS)
        
        rc = cv2.minAreaRect(contours[0])
        contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
        contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)
        
        
        # simplify contours down to polygons
        rects = []
        for cont in contours:
            rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
            rects.append(rect)

        print(len(rects))
        print(rects)

        # that's basically it
        cv2.drawContours(orig, rects,-1,(0,255,0),1)

        # show only contours
        new = get_new(img)
        cv2.drawContours(new, rects,-1,(0,255,0),1)
        cv2.GaussianBlur(new, (9,9), 0, new)
        new = cv2.Canny(new, 0, CANNY, apertureSize=3)

        cv2.imshow('Orig', orig)
        # cv2.imshow('Dilated', dilated)
        # cv2.imshow('Edges', edges)
        cv2.imshow('New', new)


        # box = cv2.boxPoints(rects)
        if(len(rects) != 0):
            for p in rects[0]:
                pt = (p[0],p[1])
                print (pt)
                cv2.circle(orig,pt,5,(200,0,0),2)
                transform_coordinates.append(list(pt))
            # sorted(transform_coordinates, key=lambda x: x[0])
            # transform_coordinates.sort(key = lambda transform_coordinates: transform_coordinates[0])
            
            transform_coordinates = sort_points(transform_coordinates)
            print(transform_coordinates)
            # del transform_coordinates[3]
            cv2.imshow("plank", orig)
            break;
        

        # time.sleep(10)
        k = cv2.waitKey(30) & 0xff
        if(k == ord('q')):
            break

    # trans_M = get_TransformMatrix(transform_coordinates)
    writer.write(cap,transform_coordinates)

# cap.release()
cv2.destroyAllWindows()
