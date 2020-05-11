# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import imutils
import numpy as np
import os.path
from imutils.video import FPS

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    correct_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            classe_id = classes.index("bottle")

            if classe_id != classId:
                continue
        
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = np.array(boxes[i])
        box[box<0] = 0
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        correct_boxes.append(box)
    
    return correct_boxes

# Process inputs
winName = 'Frame'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


flag_detection = True
flag_tracking = False

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    if flag_detection :

        # Stop the program if reached end of video
        if not hasFrame :
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        correct_boxes = postprocess(frame, outs, classes)

        # print(correct_boxes)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        cv.imshow(winName, frame)
        cv.waitKey(1)

        if correct_boxes == []:
            continue
        # Write the frame with the detection boxes
        # if (args.image):
        #     cv.imwrite(outputFile, frame.astype(np.uint8))
        # else:
        #     vid_writer.write(frame.astype(np.uint8))

        flag_detection = False
        flag_tracking = True
        initBB = tuple(correct_boxes[0])


        args = {"tracker":"csrt"}

        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv.TrackerCSRT_create,
            "kcf": cv.TrackerKCF_create,
            "boosting": cv.TrackerBoosting_create,
            "mil": cv.TrackerMIL_create,
            "tld": cv.TrackerTLD_create,
            "medianflow": cv.TrackerMedianFlow_create,
            "mosse": cv.TrackerMOSSE_create
        }

        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    elif flag_tracking:



        # frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        if initBB is not None:
            tracker.init(frame, initBB)
            initBB = None
        else:
            (success, box) = tracker.update(frame)
            fps = FPS().start()

            # check to see if the tracking was a successq
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 4)
            else :
                flag_detection = True
                flag_tracking = False
                # update the FPS counter
            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv.imshow("Frame", frame)
            key = cv.waitKey(1) & 0xFF

            # if the 's' key is selected, we are going to "select" a bounding
            # box to track

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # # if we are using a webcam, release the pointer
            # if not args.get("video", False):
            #     vs.stop()
            # # otherwise, release the file pointer
            # else:
            #     vs.release()

            # close all windows
            # cv.destroyAllWindows()

                







                
