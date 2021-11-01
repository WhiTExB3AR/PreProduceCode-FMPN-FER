# USAGE
# python haar_face_detector.py --image images/adrian_01.png

# import the necessary packages
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
parser.add_argument("-c", "--cascade", type=str,
	# default=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml",
	default=os.path.dirname(cv2.__file__)+"/data/face_detector.xml",
	help="path to haar cascade face detector")
args = vars(parser.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# load the input image from disk, resize it, and convert it to
# grayscale
image_org = cv2.imread(args["image"])
image_reszie = imutils.resize(image_org, width=500)
gray = cv2.cvtColor(image_reszie, cv2.COLOR_BGR2GRAY)

# detect faces in the input image using the haar cascade face
# detector
print("[INFO] performing face detection...")
faces = detector.detectMultiScale(
    gray, 
    scaleFactor=1.05,
	minNeighbors=7, 
    minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} faces detected...".format(len(faces)))

# loop over the bounding boxes
for (x, y, w, h) in faces:
	# draw the face bounding box on the image (B, G, R) = (255, 0, 0) => box line is blue
	cv2.rectangle(image_org, (x, y), (x + w, y + h), (255, 0, 0), 2)

# print("width: {} pixels".format(w))
# print("height: {}  pixels".format(h))

# show the output image
# cv2.imshow("Image", image)
cv2.imwrite("Face_detected.png", image_org)
print('Successfully saved and show now')
cv2.imshow("Face detected.png", image_org)
cv2.waitKey(0)
