import cv2 as cv
import numpy as np


class Tracking:
    def __init__ (self, type_tracker='csrt'):

        self.type_tracker = type_tracker

        OPENCV_OBJECT_TRACKERS = {
                                "csrt": cv.TrackerCSRT_create,
                                "kcf": cv.TrackerKCF_create,
                                "boosting": cv.TrackerBoosting_create,
                                "mil": cv.TrackerMIL_create,
                                "tld": cv.TrackerTLD_create,
                                "medianflow": cv.TrackerMedianFlow_create,
                                "mosse": cv.TrackerMOSSE_create
                           }
        self.tracker = OPENCV_OBJECT_TRACKERS[self.type_tracker]()

    def tracker_init(self, frame, initBB):
        self.tracker.init(frame, initBB)
    
    def tracker_update (self, frame):
        success, box = self.tracker.update(frame)
        return success, box
        
