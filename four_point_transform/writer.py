# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
 
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",
# 	help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=256,
# 	help="max buffer size")
# args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
##greenLower = (29, 86, 6)
##greenUpper = (64, 255, 255)
greenLower = (37,100,0)
greenUpper = (95,255,255)
pts = deque(maxlen=12800)
pts_col = deque(maxlen=12800)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[2] = pts[np.argmin(s)]
	rect[0] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[3] = pts[np.argmin(diff)]
	rect[1] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


# # if a video path was not supplied, grab the reference
# # to the webcam
# if not args.get("video", False):
# 	vs = VideoStream(src=1).start()
 
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
 
# # allow the camera or video file to warm up
# time.sleep(2.0)
def write(vs,transform_coordinates):
	# keep looping
	q_buf = [[0,0] for i in range(5)]
	q_ctr = 0
	threshold = 20
	frame = vs.read()
	canvas = np.ones_like(frame)*255
	color_mode = [0,0,255]
	while True:
		# grab the current frame
		frame = vs.read()
	

		# handle the frame from VideoCapture or VideoStream
		# frame = frame[1] if args.get("video", False) else frame
	 
		# if we are viewing a video and we did not grab a frame,
		# then we have reached the end of the video
		if frame is None:
			break

		# frame = ((frame/255.0)**2)*255
	 
		# resize the frame, blur it, and convert it to the HSV
		# color space
		# frame = imutils.resize(frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	 
		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, greenLower, greenUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		center = None
		color = None
	 
		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				# cv2.circle(frame, (int(x), int(y)), int(radius),
				# (0, 255, 255), 2)
				center = (int(x),int(y+radius))
				color = (color_mode[0], color_mode[1], color_mode[2])
				cv2.circle(frame,center , 5, (color_mode[0], color_mode[1], color_mode[2]), -1)
	 
			# update the points queue
			# cur_pos = q_ctr%(len(q_buf))
			# q_buf[cur_pos] = center
		# 
		# if(q_ctr>=len(q_buf)):
		# 	if(center==None):
		# 		continue
		# 	prev_center = q_buf[(q_ctr+1)%(len(q_buf))]
		# 	dist = ((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)**0.5
		# 	print(dist)
		# 	if(dist < threshold):
		# 		print("SIGH")
		# 		pts.appendleft(center)
		# 		q_ctr+=1
		# else:
		# 	print("HI")
		# 	pts.appendleft(center)
		# 	q_ctr+=1
		pts.appendleft(center)
		pts_col.appendleft(color)
		
		

		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue
	 
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
			thickness = 10
			print(thickness)
			cv2.line(canvas, pts[i - 1], pts[i], pts_col[i], max(thickness,1))
	 
		# show the frame to our screen
		cv2.imshow("Frame", frame)
		

		# palette = cv2.warpAffine(frame,transform_coordinates,(1280,640))

		palette = four_point_transform(canvas,np.array(transform_coordinates))
		# print(palette.shape)
		cv2.imshow("Result", palette)
	 
		# if the 'q' key is pressed, stop the loop
		key = cv2.waitKey(30) & 0xFF
		if key == ord("q"):
			break
		if key == ord(" "):
			for i in range(2):
				color_mode[i] = 255 - color_mode[i]
 
# # if we are not using a video file, stop the camera video stream
# if not args.get("video", False):
# 	vs.stop()
 
# # otherwise, release the camera
# else:
# 	vs.release()
 
# # close all windows
# cv2.destroyAllWindows()
