# https://github.com/ipazc/mtcnn

import argparse
import imutils
# imutils for crop, rotation and resize (also can transition, get edge, plat 3D to 2D and visualize in matplotlib)
# https://github.com/PyImageSearch/imutils
import cv2
import os
from PIL import Image # can use for crop image
from mtcnn import MTCNN
from pathlib import Path

detector = MTCNN()

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-img", "--image", 
                    type=str, 
                    required=True, 
                    help="path to input image")
args = vars(parser.parse_args())

# image_org = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
# faces_result = detector.detect_faces(image_org)
image_org = cv2.imread(args["image"])
gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
faces_result = detector.detect_faces(rgb)

# Result is an array with all the bounding boxes detected. For 1 face:
# bounding_box = faces_result[0]['box']
# keypoints = faces_result[0]['keypoints']

# cv2.rectangle(
#     rgb,
#     (bounding_box[0], bounding_box[1]),
#     (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#     (0,155,255),
#     2
# )

# Result is an array with all the bounding boxes detected. For many faces:
for i in range(len(faces_result)):
    bounding_box = faces_result[i]['box']
    keypoints = faces_result[i]['keypoints']

    cv2.rectangle(
        rgb,
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
        (255,155,0),
        2 # width of line of box
    )


## Show point for 5 landmark points
# cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

print("[INFO] {} faces detected...".format(len(faces_result)))
## print all result from JSON file faces_result
# print("[Face result landmark] ", faces_result) 
for i in range(len(faces_result)):
    print(faces_result[i]['box'], faces_result[i]['confidence'])

# ------- Start: split to file name -------
# 1. using pathlib
file_name_pathlib = Path(args["image"]).stem
print("Got the file name original: ", file_name_pathlib)

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
            "Face Detected MTCNN_" + 
            str(file_name_pathlib) + 
            ".png", 
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) # Convert img detected in line 28 from rgb to bgr 
print('=> Successfully saved face detection rectangle and show each face now')

# ------- Start: Crop face -------
# 1. using OpenCV
for i in range(len(faces_result)):
    bounding_box = faces_result[i]['box']
    y = bounding_box[0]
    x = bounding_box[1]
    w = bounding_box[1] + bounding_box[3]
    h = bounding_box[0] + bounding_box[2]
    crop_face = gray[x:w, y:h]
    cv2.imwrite('images/results/face_focus/mtcnn' +
                str(file_name_pathlib) +
                '_MTCNN_face_' +
                str(i) +
                '.png', 
                crop_face)
    cv2.imshow("Cropped face.png", crop_face)
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