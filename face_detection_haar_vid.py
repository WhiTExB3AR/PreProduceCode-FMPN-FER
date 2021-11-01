# # Usage
# # Now, run the project file using: python3 face_detection.py

# from PIL import Image
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np

# import cv2

# import os

# import torch
# import torchvision.transforms as transforms

# # Initialize the classifier:
# cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

# # Apply faceCascade on webcam frames
# video_capture = cv2.VideoCapture(0)
# while True:
#     # Capture frame-by-frame
#     ret, frames = video_capture.read()
#     gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     # Display the resulting frame
#     cv2.imshow('Video', frames)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture frames:
# video_capture.release()
# cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------------

# USAGE
# python video_face_detector.py

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the bounding boxes
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()