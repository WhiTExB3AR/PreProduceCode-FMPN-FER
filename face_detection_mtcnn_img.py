# https://github.com/ipazc/mtcnn

import argparse
import cv2
from mtcnn import MTCNN

detector = MTCNN()

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(parser.parse_args())

image_org = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image_org)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(
    image_org,
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

# Save img_detected
path = 'D:/OpenCV/Scripts/Images'
# cv2.imwrite("Face detected.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
img_detected = cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR)
cv2.imshow("Face detected.png", img_detected)
cv2.waitKey(0)

print(result)