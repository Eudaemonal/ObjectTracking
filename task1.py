#!/usr/bin/python3.5


import numpy as np
import cv2
import matplotlib.pyplot as plt

def auto_canny(image, sigma=0.95):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


def angle_cos(p0, p1, p2):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
	img = cv2.GaussianBlur(img, (5, 5), 0)
	squares = []
	for gray in cv2.split(img):
		for thrs in range(0, 255, 26):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 50, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				_retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
					cnt = cnt.reshape(-1, 2)
					max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
					if max_cos < 0.1:
						squares.append(cnt)
	return squares



if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')
	frameNum = 0;
	while(video.isOpened()):
		ret, frame = video.read()
		# Take first frame for matching
		if(frameNum==0):
			model = frame
			cv2.imwrite("f048.png",model)
		
		# Convert RGB to grey scale for processing
		img1 = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
		# Draw first 10 matches.

		

		img3 = cv2.drawMatches(model,kp1,frame,kp2, matches[:50],None, flags=2)

		cv2.imshow('frame',img3)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		frameNum+=1;
		print(frameNum)

	video.release()
	cv2.destroyAllWindows()
