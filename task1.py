#!/usr/bin/python3.5


import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')
	frameNum = 0;
	while(video.isOpened()):
		ret, frame = video.read()

		# Take first frame for matching
		if(frameNum==0):
			model = frame
	    
		# Initiate ORB detector
		orb = cv2.ORB_create()
		# find the keypoints and descriptors with ORB
		kp1, des1 = orb.detectAndCompute(model,None)
		kp2, des2 = orb.detectAndCompute(frame,None)
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

	video.release()
	cv2.destroyAllWindows()
