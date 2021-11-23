from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations
import sys

height1 = 0
height2 = 0

distance = 0

start_point = ()
end_point = ()

def is_close(p1, p2):
    dst = math.sqrt(p1**2 + p2**2)
    return dst 

###############################################
def closest_person(list1, ht):
    list2 = list1.copy()
    list3 = []
    list4 = []
    for i in range(len(list1)):
        list3 = []
        for j in range(len(list2)):
            pt1 = list1[i]
            pt2 = list2[j]
            dx = (pt1[0] - pt2[0])
            dy = (pt1[1] - pt2[1])

            distance = is_close(dx,dy)
            list3.append(distance)
            
        list4.append(list3)
    
    minvalue = []
    index = []
    for i in range(len(list4)):
        if any(j > 0 for j in list4[i]):
            minvalue_ = min(j for j in list4[i] if j > 0)   
            if minvalue_ != None:
                minvalue.append(minvalue_)
                index.append(list4[i].index(minvalue_))
    return index

def closest_value(list1, ht):
    list2 = list1.copy()
    list3 = []
    list4 = []
    for i in range(len(list1)):
        list3 = []
        for j in range(len(list2)):
            pt1 = list1[i]
            pt2 = list2[j]
            dx = (pt1[0] - pt2[0])
            dy = (pt1[1] - pt2[1])

            distance = is_close(dx,dy)
            list3.append(distance)
            
        list4.append(list3)
    
    minvalue = []
    index = []
    for i in range(len(list4)):
        if any(j > 0 for j in list4[i]):
            minvalue_ = min(j for j in list4[i] if j > 0)   
            if minvalue_ != None:
                minvalue.append(minvalue_)
                index.append(list4[i].index(minvalue_))
    return minvalue

##################################################

def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        for detection in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   # Coco file has string of all the names
            if name_tag == 'person':                
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# Store the center points of the detections
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
                # Append center point of bbox for persons detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
                objectId += 1 #Increment the index for each detection      

		#=================================================================
    	# 3.3 Purpose : Display closest Detection
    	#=================================================================  
        
        for id_, box_ in centroid_dict.items():
            height1 = (box_[5] - box_[3])

            closest = closest_person(list(centroid_dict.values()), height1)
            closest_dist = closest_value(list(centroid_dict.values()), height1)

            if closest != [] and closest_dist != [] and (box_[4] - box_[2]) < 200:

                start_point = (box_[0],box_[1])
                end_point = (list(centroid_dict.values())[closest[id_]][0],list(centroid_dict.values())[closest[id_]][1])

                cv2.rectangle(img, (box_[2], box_[3]), (box_[4], box_[5]), (0, 0, 255), 1) # Create Red bounding boxes  #starting point, ending point size of 2
                cv2.putText(img, str(id_) + ' > ' + str(closest[id_]) + ' [' + str(round(((closest_dist[id_])*4)/height1, 2)) + ']', 
                        (box_[0], box_[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 255], 1)
                
                distance = (round(((closest_dist[id_])*4)/height1, 2))

                if distance < 6:
                    cv2.rectangle(img, (box_[2], box_[3]), (box_[4], box_[5]), (255, 0, 0), 1)
                    cv2.putText(img, 'danger', (box_[2], box_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        [255, 0, 0], 1)
                    cv2.line(img, start_point, end_point, (255, 0, 0), 1)

    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./paths/yolov4.cfg"
    weightPath = "./paths/yolov4.weights"
    metaPath = "./paths/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(sys.argv[1])
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (new_width, new_height))

    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
