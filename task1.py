#!/usr/bin/python3.5


import numpy as np
import cv2
import time


if __name__ == "__main__":
	video = cv2.VideoCapture('Video_sample.mp4')

	while(video.isOpened()):
	    ret, frame = video.read()

	    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	video.release()
	cv2.destroyAllWindows()