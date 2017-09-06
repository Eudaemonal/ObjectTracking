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


def detect_item(img, show_contour, show_center):
	num_item = 0
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)

	gray = cv2.GaussianBlur(gray,(5,5),0)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 375, 58)

	edge = auto_canny(thresh)

	(_, cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)

		if M['m00'] != 0:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			area = cv2.contourArea(c)

			if (24<area) & (area < 40960):
				num_item+=1
				#print("(%3d, %3d): %3d"%(cx, cy, area))
				if(show_center):
					cv2.circle(img,(cx,cy), 1, (0,0,255), 6)
					location = 'Item #'+str(num_item)+' '+'('+str(cx)+', '+str(cy)+')'
					cv2.putText(img,location,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)

				if(show_contour):
					cv2.drawContours(img, c, -1, (0, 255,255), 3)

	return img, num_item


if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')

	frameNum = 0
	fps = video.get(cv2.CAP_PROP_FPS)

	while(video.isOpened()):
		ret, frame = video.read()
		# Take first frame for matching

		if(frameNum==0):
			model = frame
		
		# Convert RGB to grey scale for processing
		img1 = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame, num_item = detect_item(frame, True, True)
		#print("Num: %d"%(num_item))

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
