# Author: Thejus Singh Jagadish
# Date Created: 11/ 12/2017



from imutils.video import VideoStream
import RPi.GPIO as GPIO
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

from ubidots import ApiClient
import math

def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	C = euclidean_dist(eye[0], eye[3])
	ear = (A+B)/(2.0*C)
	return ear

# Create an ApiClient object

api = ApiClient(token='A1E-DA86a1Tq3hMsOYodDdnj2c5F0EJdY3')

# Get a Ubidots Variable

variable = api.get_variable('5a025bd6c03f97318dff50f7')

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True)
ap.add_argument("-p", "--shape-predictor", required=True)
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 5

COUNTER = 0
SLEEPCOUNTER = 0

detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src = 0).start()
time.sleep(1.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=570)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in rects:
		rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR)/2.0
		print(ear)
		response = variable.save_value({"value": ear})

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				SLEEPCOUNTER += 1
				GPIO.output(18, True)
		else:
			GPIO.output(18, False)
			COUNTER = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		GPIO.output(18, False)
		break

cv2.destroyAllWindows()
vs.stop()

