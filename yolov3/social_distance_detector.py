
MODEL_PATH = "yolo-coco"
MIN_CONF = 0.3
NMS_THRESH = 0.4
USE_GPU = False
MIN_DISTANCE = 100



from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

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
	print(len(results))

	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				print(D[i, j])
				if D[i, j] < MIN_DISTANCE:
					print(D[i, j])
					print("[INFO] tá entrando aqui")
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

	# # _____________________________________________________________________#

    #         (mask, without_mask) = maskNet.predict(face)[0]
    #         label = "Mask" if mask > without_mask else "No Mask"
    #         color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

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


	if writer is not None:
		writer.write(frame)