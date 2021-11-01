# https://github.com/ipazc/mtcnn

import argparse
import imutils
import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-img", "--image", type=str, required=True, help="path to input image")
args = vars(parser.parse_args())

# image_org = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
# faces_result = detector.detect_faces(image_org)
image_org = cv2.imread(args["image"])
rgb = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
faces_result = detector.detect_faces(rgb)

# Result is an array with all the bounding boxes detected.
bounding_box = faces_result[0]['box']
keypoints = faces_result[0]['keypoints']

cv2.rectangle(
    rgb,
    (bounding_box[0], bounding_box[1]),
    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
    (0,155,255),
    2
)

## Show point for 5 landmark points
# cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

print("[Face result landmark] ", faces_result)

# Save img_detected
img_detected = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_detected, cv2.COLOR_BGR2GRAY)

# path = 'E:\THESIS\GITHUB\PreProduceCode-FMPN-FER\images\results\face_detected'
# cv2.imwrite(os.path.join(path, "Face detected.png", img_detected))
print('=> Successfully saved and show now')
cv2.imshow("Face detected MTCNN.png", img_detected)
cv2.waitKey(0)