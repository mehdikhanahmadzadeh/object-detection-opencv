import cv2 as cv
import numpy as np
import sys

class Detection:
	def __init__ (self, modelConfiguration='', modelWeights='',
					classesFile='', confThreshold=0.5, nmsThreshold=0.4,
					inpWidth=416, inpHeight=416 ) :
		self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
		self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
		self.classesFile = classesFile
		self.classes = None
		self.confThreshold = confThreshold  
		self.nmsThreshold = nmsThreshold  
		self.inpWidth = inpWidth       
		self.inpHeight = inpHeight      

	def load_class_name(self):
		with open(self.classesFile, 'rt') as f:
			self.classes = f.read().rstrip('\n').split('\n')

	def postprocess(self, frame, outs, classes):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]

		classIds = []
		confidences = []
		boxes = []
		correct_boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				classId = np.argmax(scores)
				confidence = scores[classId]
				classe_id = self.classes.index("bottle")

				if classe_id != classId:
					continue
			
				if confidence > self.confThreshold:
					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)
					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

		indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
		for i in indices:
			i = i[0]
			box = np.array(boxes[i])
			box[box<0] = 0
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
			correct_boxes.append(box)
		
		return correct_boxes

	def getOutputsNames(self):
		layersNames = self.net.getLayerNames()
		return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def drawPred(self, frame, classId, conf, left, top, right, bottom):
		cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
		label = '%.2f' % conf
		if self.classes:
			assert(classId < len(self.classes))
			label = '%s:%s' % (self.classes[classId], label)
			labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			top = max(top, labelSize[1])
			cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
			cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)





