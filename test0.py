#!/usr/bin/python3.5

from shapedetector import ShapeDetector
import argparse
import imutils
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




if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')


	while(video.isOpened()):
		ret, img = video.read()
		mask = np.zeros_like(img)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)

		thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
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

				if (32<area) & (area < 40960):
					print("(%3d, %3d): %3d"%(cx, cy, area))
					cv2.circle(img,(cx,cy), 1, (0,0,255), 6)
					location = '('+str(cx)+', '+str(cy)+')'
					cv2.putText(img,location,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)

		cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
		cv2.imshow("thresh", img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video.release()
	cv2.destroyAllWindows()