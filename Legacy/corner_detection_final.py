import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
  
  
# capture frames from a camera 
cap = cv2.VideoCapture(1) 
  

# loop runs if capturing has been initialized 
while(1): 
  
	# reads frames from a camera 
	ret, frame = cap.read() 
	
	# print(frame.shape)
	# break
	# converting BGR to HSV 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
	  
	# define range of red color in HSV 
	lower_red = np.array([30,150,50]) 
	upper_red = np.array([255,255,180]) 
	  
	# create a red HSV colour boundary and  
	# threshold HSV image 
	mask = cv2.inRange(hsv, lower_red, upper_red) 
  
	# Bitwise-AND mask and original image 
	res = cv2.bitwise_and(frame,frame, mask= mask) 
  
	# Display an original image 
	cv2.imshow('Original',frame) 
  
	# finds edges in the input image image and 
	# marks them in the output map edges 
	edges = cv2.Canny(frame,100,200) 

	# Display edges in a frame 
	cv2.imshow('Edges',edges)

	new = np.zeros_like(edges)

	# new[:,:int(edges.shape[1]*.8)] = edges[:,:int(edges.shape[1]*.8)]
	new = edges.copy()
	prev_20 = [0 for i in range(20)]
	ctr = 0

	x,y = [-1,-1,-1,-1],[-1,-1,-1,-1]
	# print(edges.shape)
	# break
	for i in range(int(new.shape[0]/2),new.shape[0]):
		y_min = -1
		y_max = new.shape[0]
		for j in range(0,new.shape[1]):
			if new[i][j] != 0:
				if y_min == -1:
					y_min = j
				else:
					y_max = j
		prev_20[ctr%20]=y_max-y_min
		ctr+=1
		if ctr < 20:
			continue
		# print((prev_20[(ctr+1)%20]-prev_20[ctr%20]) / prev_20[(ctr+1)%20])
		
		if(prev_20[(ctr+1)%20]-prev_20[ctr%20]>.09*prev_20[(ctr+1)%20]):
			if(x[1]==-1):
				x[1]=i
				if y_min == -1:
					print("assigning -1")
					break
				y[1]=y_min

		if(prev_20[ctr%20]<10):
			# if(x[3]==-1):
			x[3]=i
			y[3]=y_max
			break

	ctr = 0
	for i in range(0,int(new.shape[0]/2)):
		y_min = -1
		y_max = new.shape[0]
		for j in range(0,new.shape[1]):
			if new[i][j] != 0: 
				if y_min == -1:
					y_min = j
				else:
					y_max = j
		prev_20[ctr%20]=y_max-y_min
		# print(prev_20[ctr%20])
		ctr+=1
		if ctr<20:
			continue
		if(prev_20[(ctr+1)%20]-prev_20[ctr%20]>.09*prev_20[(ctr+1)%20]):
			if(x[0]==-1):
				x[0]=i
				y[0]=y_min

		if(prev_20[ctr%20]<10):
			# if(x[2]==-1):
			# print("entering here, about to break",i,y_max)
			x[2]=i
			y[2]=y_max
			break

	print(x,y)
	new2 = np.zeros_like(edges)
	new2[x,y] = 255
	# c = edges.copy()
	# extLeft = tuple(c[c[:, :, 0].argmin()][0])
	# extRight = tuple(c[c[:, :, 0].argmax()][0])
	# print(extLeft,extRight)
	# extTop = tuple(c[c[:, :, 1].argmin()][0])
	# extBot = tuple(c[c[:, :, 1].argmax()][0])
	cv2.imshow('new2',new2)

	# for i in range(10):
	# 	ith, jth = np.unravel_index(dst.argmax(),dst.shape)
	# 	if(dst[ith,jth]>0):
	# 		dst[ith,jth] = 0
	# 		new[ith,jth] = 255
	# 	else:
	# 		break

	cv2.imshow('New',new)

	k = cv2.waitKey(5) & 0xFF
	if k == 27: 
		break


# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 