# USAGE
# python haar_face_detector.py --image images/adrian_01.png

# import the necessary packages
import argparse
import imutils
import cv2
import os

from PIL import Image # can use for crop image
from pathlib import Path


# construct the argument parser and parse the arguments
haar_fontface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", 
    type = str, 
    required = True, 
    help = "path to input image")
parser.add_argument("-c", "--cascade", 
    type = str, 
    default = haar_fontface, 
    help = "path to haar cascade face detector")
args = vars(parser.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# load the input image from disk, resize it, and convert it to
# grayscale
image_org = cv2.imread(args["image"])
# image_resize = imutils.resize(image_org, width=500)
gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)

# detect faces in the input image using the haar cascade face
# detector
print("[INFO] performing face detection...")
faces_result = detector.detectMultiScale(
    gray, 
    scaleFactor = 1.05,
	minNeighbors = 7, 
    minSize = (30, 30),
	flags = cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} faces detected...".format(len(faces_result)))
count = 0
for i in range(len(faces_result)):
    count = count + 1


# loop over the bounding boxes
for (x, y, w, h) in faces_result:
	# draw the face bounding box on the image (B, G, R) = (255, 0, 0) => box line is blue
	cv2.rectangle(image_org, (x, y), (x + w, y + h), (0,155,255), 2)

# print("width: {} pixels".format(w))
# print("height: {}  pixels".format(h))

# ------- Start: split to file name -------
# 1. using pathlib
file_name_pathlib = Path(args["image"]).stem
print(file_name_pathlib)

# # 2. using os module basename and splitext
# base_name = os.path.basename(args["image"])
# file_name = os.path.splitext(base_name)[0]
# print(file_name)

# # 3. using os module basename and split
# file_name_spilt = os.path.basename(args["image"]).split('.')[0]
# print(file_name_spilt)

# # 4. using split
# directory = args["image"]
# name = directory.split('.')
# filename = name[0].split('/')
# print(filename[-1])
# ------- End: split to file name -------

# Save the image with rectangle face detection
cv2.imwrite('images/results/detected_rectangle/' + 
            "Face Detected_" + 
            str(file_name_pathlib) + 
            ".png", 
            image_org)
print('=> Successfully saved face detection rectangle and show each face now')

# ------- Start: Crop face -------
# 1. using OpenCV
i = 0 # for each face focus
for (x, y, w, h) in faces_result:
    print(x,y,w,h)
    crop_face = gray[y:y+h, x:x+w]
    cv2.imwrite('images/results/face_focus/' +
                str(file_name_pathlib) +
                'HAAR_face_' +
                str(i) +
                '.png', 
                crop_face)
    cv2.imshow("Cropped face.png", crop_face)
    count = count - 1
    if (count < 0): # no face focus found in len(faces_result)
        break
    cv2.waitKey(0)

# # 2. using PIL
# for i in range(len(faces_result)):
#     bounding_box = faces_result[i]['box']
#     left = bounding_box[0]
#     right = bounding_box[1]
#     top = bounding_box[1] + bounding_box[3]
#     bottom = bounding_box[0] + bounding_box[2]
#     img_pil = Image.open(args["image"])
#     crop_image = img_pil.crop((left, top, right, bottom))
#     # cv2.imwrite("Face detected",[i],".png", crop_image)
#     crop_image.show()
#     cv2.waitKey(0)
# ------- End: Crop face -------

# show the output image
# cv2.imshow("Image", image)
# path = 'images/results/face_detected'
# cv2.imwrite(os.path.join(path,"Face detected Haar.png", image_org))
# print('Successfully saved and show now')
# cv2.imshow("Face detected Haar.png", image_org)
# cv2.waitKey(0)