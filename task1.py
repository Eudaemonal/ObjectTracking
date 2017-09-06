#!/usr/bin/python3.5


import numpy as np
import cv2
import matplotlib.pyplot as plt



def edgeDetect (img, ksize = 5):
	sobelX  = cv2.Sobel(img,cv2.CV_16S,1,0,ksize)
	sobelY  = cv2.Sobel(img,cv2.CV_16S,0,1,ksize)
	sobel = np.hypot(sobelX, sobelY)/255
	sobel[sobel > 1] = 1

	return sobel

def findSignificantContours (img, edge):
	num_item = 0
	(_, cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_LIST ,
			cv2.CHAIN_APPROX_SIMPLE)
	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)

		if M['m00'] != 0:
			
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			area = cv2.contourArea(c)

			# opal: 17900, brush: 24000
			if (9400<area) & (area < 25000):
				num_item+=1
				# Show centre of item
				cv2.circle(img,(cx,cy), 1, (0,0,255), 6)
				location = 'Item #'+str(num_item)+' '+'('+str(cx)+', '+str(cy)+')'
				cv2.putText(img,location,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
				# Show contour of item
				cv2.drawContours(img, c, -1, (0, 255,255), 2, cv2.LINE_AA,maxLevel=1)

	return img


def detect_item(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)

	blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
	edgeImg = np.max( np.array([ edgeDetect(blurred[:,:, 0]), edgeDetect(blurred[:,:, 1]), edgeDetect(blurred[:,:, 2]) ]), axis=0 )
	edge= np.array(edgeImg * 255, dtype = np.uint8)
	mean = np.mean(edge);
	# Zero any value that is less than mean. This reduces a lot of noise.
	edge[edge <= mean] = 0


	thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)
	thresh = cv2.add(thresh2, edge)

	img = findSignificantContours(img, edge)

	return img


if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')
	ret, model = video.read()
	h, w, d = model.shape

	frameNum = 1

	while(video.isOpened()):
		ret, frame = video.read()
		# Take first frame for matching
		
		# Convert RGB to grey scale for processing
		img1 = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame = detect_item(frame)


		# Initiate ORB detector
		orb = cv2.ORB_create()
		# find the keypoints and descriptors with ORB
		kp1, des1 = orb.detectAndCompute(img1,None)
		kp2, des2 = orb.detectAndCompute(img2,None)
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# Match descriptors.
		matches = bf.match(des1,des2)
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		
		img3 = cv2.drawMatches(model,kp1,frame,kp2, matches[:48],None, flags=2)

		cv2.imshow('frame',img3)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		frameNum+=1;

	video.release()
	cv2.destroyAllWindows()
