
import cv2 as cv
import argparse
import sys
import imutils
import numpy as np
import os.path
from imutils.video import FPS
from detection_OOP import Detection
from tracking_OOP import Tracking

winName = 'Frame'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture(0)

flag_detection = True
flag_tracking = False

detector = Detection(
                modelConfiguration="yolov3-tiny.cfg",
                modelWeights="yolov3-tiny.weights",
                classesFile="coco.names")
classes = detector.load_class_name()
# outs = detector.net.forward(detector.getOutputsNames())

while cv.waitKey(1) < 0:
    

    hasFrame, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    if flag_detection :


        if not hasFrame :
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (detector.inpWidth, detector.inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        detector.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = detector.net.forward(detector.getOutputsNames())

        # Remove the bounding boxes with low confidence
        correct_boxes = detector.postprocess(frame, outs, classes)

        # print(correct_boxes)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = detector.net.getPerfProfile()
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
        initBB = tuple(correct_boxes[0])
        tracker = Tracking(type_tracker='csrt')
        # tracker.tracker_init(frame, initBB)
        flag_detection = False
        flag_tracking = True

    elif flag_tracking:

        (H, W) = frame.shape[:2]

        if initBB is not None:
            tracker.tracker_init(frame, initBB)
            initBB = None
        else:
            success, box = tracker.tracker_update(frame)
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
                ("Tracker", tracker.type_tracker),
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

                







                
