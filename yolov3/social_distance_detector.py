
MODEL_PATH = "yolo-coco"
MIN_CONF = 0.3
NMS_THRESH = 0.4
USE_GPU = False
MIN_DISTANCE = 50



from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import time
import cv2
import os

start_time = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=0,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


if USE_GPU:
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
while True:
	(grabbed, frame) = vs.read()


	if not grabbed:
		break

	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))
	
	violate = set()
	abnormal = set()

	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < MIN_DISTANCE:
					violate.add(i)
					violate.add(j)



	for (i, (prob, bbox, centroid)) in enumerate(results):

		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		if i in violate:
			color = (0, 0, 255)

		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)


	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 55),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

	if args["output"] != "" and writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 2,
			(frame.shape[1], frame.shape[0]), True)

		# if args["input"]:
		# 		video = cv2.VideoCapture("teste2.avi");
		# 		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

		# 		if int(major_ver)  < 3 :
		# 			fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
		# 			print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
		# 		else :
		# 			fps = video.get(cv2.CAP_PROP_FPS)
		# 			print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

		# video.release()

		# if not args["input"]:
		# # Start default camera
		# 	video = cv2.VideoCapture(0);

		# 	# Find OpenCV version
		# 	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

		# 	# With webcam get(CV_CAP_PROP_FPS) does not work.
		# 	# Let's see for ourselves.

		# 	if int(major_ver)  < 3 :
		# 		fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
		# 		print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
		# 	else :
		# 		fps = video.get(cv2.CAP_PROP_FPS)
		# 		print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

		# 	# Number of frames to capture
		# 	num_frames = 120;

		# 	print("Capturing {0} frames".format(num_frames))

		# 	# Start time
		# 	start = time.time()

		# 	# Grab a few frames
		# 	for i in range(0, num_frames) :
		# 		ret, frame = video.read()

		# 	# End time
		# 	end = time.time()

		# 	# Time elapsed
		# 	seconds = end - start
		# 	print ("Time taken : {0} seconds".format(seconds))

		# 	# Calculate frames per second
		# 	fps  = num_frames / seconds
		# 	print("Estimated frames per second : {0}".format(fps))

		# 	# Release video
		# 	video.release()


	
	if writer is not None:
		writer.write(frame)
