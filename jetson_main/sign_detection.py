import serial
from adafruit_servokit import ServoKit
import time
import os
import argparse
import cv2
import numpy as np
from threading import Thread
import importlib.util
import websockets
import asyncio
import json

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
MODEL_NAME = 'TFLite_model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.93
resH = '720'
resW = '1280'
imW, imH = int(resW), int(resH)
use_TPU = False

def predictLeNet(image):
    model = 'TFLite_model_recognizeLeNet/recognizeLeNet_model.tflite'
    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    image = np.array(cv2.resize(image, (32, 32)), dtype=np.float32)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    image = image.reshape(1, 32, 32, 3)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    labels = {0: "No sign", 1: "prohibited", 2: "stop", 3: "left", 4: "right",
              5: "30", 6: "50", 7: "80"}
    label_to_probabilities = []
    size = len(labels)
    for i, probability in enumerate(probabilities):
        if i > len(labels) - 1:
            break
        label_to_probabilities.append(float(probability))

    sign = labels[label_to_probabilities.index(max(label_to_probabilities))]
    value = max(label_to_probabilities)
    print("value: " + str(value))
    if value > 0.70:
        result = '%s: %.2f%%' % (sign, (100 * value))
    else:
        result = "No sign"

    return result
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')

from tflite_runtime.interpreter import Interpreter
if use_TPU:
    from tflite_runtime.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

#if __name__ == '__main__':
async def recordVideo(websocket):
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)
    distanceInitiale = 60
    tailleInitiale = 188
    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        dataSignDistance = []
        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                tailleMesure = ymax - ymin
                distanceActuelle = 2*distanceInitiale - ((tailleMesure*distanceInitiale)/tailleInitiale)
                #print(distanceActuelle)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
                cropped = frame[ymin:ymax, xmin:xmax]
                prediction = predictLeNet(cropped)
                sign = prediction.split(':')
                #print(sign[0])
                if sign[0] != 'No sign':
                    dataSignDistance.append([sign[0],distanceActuelle,True])

                #print(dataSignDistance)
                #print(prediction)
                label = '%s ' % (prediction)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        man = cv2.imencode('.jpg', frame, encode_param)[1]
        #await asyncio.sleep(0.033)
        await websocket.send(man.tobytes())
        cv2.imshow('Object detector', frame)
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq 
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        
        if len(dataSignDistance) > 0:
            sign = {
                    "sign": dataSignDistance[0][0],
                    "distance": dataSignDistance[0][1],
                    "detected": dataSignDistance[0][2]
                    }
        else :
            sign = {
                    "sign" : "nothing",
                    "distance" : 0,
                    "detected" : True
                    }

        with open('signs.json','w') as data:
            json.dump(sign,data)

ipServeur = "10.30.50.185" #Adresse ISEN Project
connexionServeur = websockets.serve(recordVideo, ipServeur, 5678)
asyncio.get_event_loop().run_until_complete(connexionServeur)
asyncio.get_event_loop().run_forever()