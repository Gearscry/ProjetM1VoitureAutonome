import threading
import time
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

dataSignDistance = []
dataSignDistance.append(["nothing",0,True])
mode = "autonomous"
parking_mode = "off"

#-------------SIGN DETECTION-------------#
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

class SignDetection(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ipServeur = "10.30.50.186"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connexionServeur = websockets.serve(self.detection, ipServeur, 5678)
        asyncio.get_event_loop().run_until_complete(connexionServeur)
        asyncio.get_event_loop().run_forever()

    async def detection(self,websocket):

        global dataSignDistance

        #websocket = self.websocket
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
                    
                    print(dataSignDistance)
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


#------------ALL FUNCTIONNALITIES------------#
#Fonctions
def accelerationFunction(distance_front,targetedSpeed,objectDetected,initialDistance): #Fonction d'accélération et d'anticollision.
    
    v_max = 3.6

    sensorDistance = 12 #Distance qui sépare le devant de la maquette et le capteur.
    criticalDistance = 35*(targetedSpeed/v_max) #Distance critique à laquelle s'arrêter.
    if initialDistance == 0:
        initialDistance = 1
    
    distance = distance_front - sensorDistance - criticalDistance #Distance à parcourir pour s'arrêter.
    print("Distance mesurée devant : " + str(distance))
    if distance > 60 and objectDetected == True: #Si un objet est à une distance suppérieure à 40cm et qu'il était précédemment détecté.
        objectDetected = False #Il n'est plus considéré comme détecté.
        initialDistance = 0 #Remise à 0 de la distance initiale.
    elif distance <= 60 and objectDetected == False: #Si un objet est à une distance inférieure à 40cm et qu'il n'était pas précédemment détecté.
        objectDetected = True #Il est considéré comme détecté.
        initialDistance = distance #Enregistrement de sa distance comme distance initiale.
    
    if objectDetected == True: #Si un objet est détecté.
        print("Obstacle détecté à une distance de " + str(distance_front) + "cm.")
        if distance > 0: #S'il est à une distance suppérieure à la distance critique.
            consigne = 100 + (targetedSpeed/v_max)*(distance/initialDistance)*35 #Il ralentis.
        else : #S'il est à une distancei inférieure à la distance critique.
            consigne = 92 #Mise des gaz à 0.
    else : #Si aucun objet n'est détecté.
        consigne = 100 + (targetedSpeed/v_max)*35 #La consigne de la voiture est normale.

    if consigne > 180 :
        consigne = 135
    elif consigne < 0 :
        consigne = 45
    real_speed = round(((consigne-100)/35)*v_max,1)
    if real_speed < 0:
        real_speed = 0
    return consigne,objectDetected,initialDistance,real_speed

#Fonction utilitaire pour le stationnement.
def step_timer(duration,parking_step,pause,measuring,timer_start):
    print("Parking step : " + str(parking_step) + ".")
    if measuring == False:
        timer_start = time.time()
        measuring = True
    else :
        timer_end = time.time()
        time_elapsed = timer_end - timer_start - pause
        print(time_elapsed)
        if time_elapsed >= duration:
            print("Temps dépassé")
            measuring = False
            parking_step = parking_step + 1
    return parking_step,measuring,timer_start

class Mode(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ipServeur = '10.30.50.186'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connexionServeur = websockets.serve(self.modeFunction,ipServeur,5679)
        asyncio.get_event_loop().run_until_complete(connexionServeur)
        asyncio.get_event_loop().run_forever()

    async def modeFunction(self,websocket):
        
        global mode
        global parking_mode

        msg = await websocket.recv()
        if msg == "manual":
            mode = "manual"
        elif msg == "autonomous":
            mode = "autonomous"
        elif msg == "Park mode : on":
            parking_mode = "on"
        elif msg == "Park mode : off":
            parking_mode = "off"

class AllFunctionnalities(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ipServeur = "10.30.50.186"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connexionServeur = websockets.serve(self.functionnalities, ipServeur, 5680)
        asyncio.get_event_loop().run_until_complete(connexionServeur)
        asyncio.get_event_loop().run_forever()

    async def functionnalities(self,websocket):
        
        global dataSignDistance
        global mode
        global parking_mode

         #Initialisation de la connexion en Série.
        ser = serial.Serial('/dev/ttyUSB0',9600,timeout=1) #Connexion en série à la carte arduino.
        ser.reset_input_buffer()
        ser.stopbits = 2

        #Initialisation des variables.
        stopped = False
        timer_start = 0
        timer_end = 0
        targeted_speed = 3.6
        previous_speed = 0
        timer = time.time()
        previous_signs = []
        signs = []
        turning = False
        parking = False
        parked = False
        measuring = False
        parking_step = 0
        back_obstacle = False
        pause_start = 0
        pause_end = 0
        pause = 0
        previous_mode = "autonomous"
        front_distance = 400
        right_distance = 400
        back_distance = 400
        left_distance = 400
        spot_finded = False
        objectDetected = False
        initialDistance = 0
        real_speed = 3.6

        #Initialisation des constantes.
        v_max = 3.6 #Vitesse maximale en km/h.
        acceleration = 92 #Neutre de l'accélération.
        direction = 90 #Neutre de la direction.

        #Initialisation des servomoteurs.
        servos = ServoKit(channels=16)
        servos.servo[0].angle = 90 #Mode automatique.
        servos.servo[1].angle = acceleration #Accélération.
        servos.servo[2].angle = direction #Direction.

        while True:

            ###----PILOTAGE DE LA MAQUETTE----###
            print("Mode : " + str(mode) + ", parking : " + str(parking_mode))

            if parking_mode == "on":
                parking = True
            elif parking_mode == "off":
                parking = False

            if mode == "autonomous":
                if previous_mode == "manual":
                    servos.servo[0].angle = 90
                    previous_mode = "autonomous"
                    parking = False
                    parked = False
                    spot_finded = False

                line = ser.readline().decode('ISO-8859-1').rstrip()
                #print(line)
                string = line.split(",")
                #print("Distances : " + str(line))
                if len(string) > 1 and len(string[0]) <= 3 and string[0] != '':
                    front_distance = int(string[0])
                    right_distance = int(string[3])
                    back_distance = int(string[2])-11
                    left_distance = int(string[1])

                if stopped == False: #Si la voiture n'est pas arrêtée à un stop.
                    #print("Voiture pas arrêtée à un stop.") 
                    signs = dataSignDistance

                    if parked == False:
                        print("Voiture non garée.")
                        if turning == False:
                            print("Voiture pas en train de tourner")
                            if parking == True:
                                print("Voiture en mode parking")
                                if spot_finded == False:  
                                    print("Place toujours pas trouvée.")
                                    print(str(right_distance) + "cm à droite")
                                    if right_distance >= 40 and measuring == False:
                                        measuring = True
                                        timer_start = time.time()
                                        print("Measuring, right distance : " + str(right_distance) + ".")
                                    elif right_distance < 40 and measuring == True:
                                        timer_end = time.time()
                                        spot_length = 100*(targeted_speed/3.6)*(timer_end-timer_start)
                                        measuring = False
                                        print("End of measure, spot length : " + str(spot_length) + ", time elapsed : " + str(timer_end-timer_start))
                                        if spot_length > 40: #sqrt((pow(99,2)-pow(19,2))+pow(19,2)-pow((sqrt(pow(99,2)-pow(19,2))-40),2))-19
                                            spot_finded = True
                                            print("Spot finded")
                    
                                elif spot_finded == True:
                                    print("Place trouvée")
                                    if int(string[2]) <= 8 and back_obstacle == False:
                                        pause = 0
                                        back_obstacle = True
                                        pause_start = time.time()
                                    elif int(string[2]) > 8 and back_obstacle == True:
                                        back_obstacle = False
                                        pause_end = time.time()
                                        pause = pause_end - pause_start

                                    if back_obstacle == False:
                                        print("Pas d'obstacle derrière")
                                        if parking_step == 0:
                                            #On continue de faire avancer la voiture jusqu'à ce que sont arrière soit à ras de l'obstacle (environs 17 cm)
                                            print("Step 0")
                                            acceleration = 120
                                            direction = 180
                                            parking_step,measuring,timer_start = step_timer(0.1,0,pause,measuring,timer_start)
                                        elif parking_step == 1:
                                            #Etape 1 : Placement à 45 degrés
                                            print("Step 1")
                                            direction = 0
                                            acceleration = 65    
                                            parking_step,measuring,timer_start = step_timer(1.2,1,pause,measuring,timer_start)
                                        elif parking_step == 2:
                                            #Etape 2 : Déplacement vers l'arrière roues droites
                                            print("Step 2")
                                            direction = 90
                                            acceleration = 65
                                            parking_step,measuring,timer_start = step_timer(0.5,2,pause,measuring,timer_start)
                                        elif parking_step == 3:
                                            #Etape 3 : Placement droit
                                            print("Step 3")
                                            direction = 180    
                                            parking_step,measuring,timer_start = step_timer(1.2,3,pause,measuring,timer_start)
                                        elif parking_step == 4:
                                            #Etape 4 : Replacement
                                            print("Step 4")
                                            direction = 90
                                            acceleration = 115
                                            if int(string[0]) <= 20 :
                                                acceleration = 92
                                                parked = True
                                                measuring = False
                                                timer_start = 0
                                    else:
                                        print("Obstacle derrière")
                                        acceleration = 92

                            if spot_finded == False:

                                signs = dataSignDistance

                                if len(previous_signs) > 0: #Si des panneaux étaient détectés précédemment.
                
                                    if len(signs) == 0 and previous_signs[0][1] > 0: #Si il n'y a plus de panneaux mais que le panneau précédent était à plus de 60 cm.
                                        signs.append([previous_signs[0][0],previous_signs[0][1],False]) #Le panneau est toujours la mais n'est plus détecté.
                                        #print("Plus de panneaux mais le panneau précédent était à plus de 60cm.")
                                    elif len(signs) > 0 and signs[0][0] != previous_signs[0][0] and previous_signs[0][1] > 25: #Si le nouveau panneau détecté le plus proche est différent de l'ancien.
                                        temp = signs
                                        signs = [[previous_signs[0][0],previous_signs[0][1],False],temp]
                                    #print("Nouveau panneau détecté mais il est différent de l'ancien et l'ancien était à une distance suppérieure à 60cm.")

                                if len(signs) > 0:
                                    if signs[0][2] == False: #Si le panneau n'est plus détecté mais est toujours présent.
                                        signs[0][1] = signs[0][1] - 100 * (targeted_speed/v_max)*(time.time()-timer) #Modification de la distance estimée du panneau qui n'est plus détecté.
                                    #print("Le panneau " + signs[0][0] + " n'est plus détecté mais il est toujours à une distance de " + str(signs[0][1]) + "cms.")

                                    sign_name = signs[0][0] #Nom du panneau.
                                    sign_distance = signs[0][1] #Distance du panneau.

                                    if sign_name == "stop" :
                                        if sign_distance <= 25:
                                            previous_speed = targeted_speed
                                            targeted_speed = 0
                                            timer_start = time.time()
                                            stopped = True
                                            previous_signs = []
                                            signs = []
                                    elif sign_name == "prohibited":
                                        if sign_distance <= 20:
                                            targeted_speed = 0
                                            previous_signs = []
                                            print("Arrêt sens interdit")
                                            signs = []
                                    elif sign_name == "30":
                                        if sign_distance <= 5:
                                            targeted_speed = (30/80)*v_max
                                            previous_signs = []
                                            print("Passage vitesse à 30")
                                            signs = []
                                    elif sign_name == "50":
                                        if sign_distance <= 5:
                                            targeted_speed = (50/80*v_max)
                                            previous_signs = []
                                            print("Passage vitesse à 50")
                                            signs = []
                                    elif sign_name == "80":
                                        if sign_distance <= 5:
                                            targeted_speed = v_max
                                            previous_signs = []
                                            print("Passage vitesse à 80")
                                            signs = []
                                    elif sign_name == "left":
                                        if sign_distance <= 20:
                                            direction = 180
                                            previous_signs = []
                                            turning = True
                                            timer_start = time.time()
                                            print("Tourne à gauche")
                                            signs = []
                                    elif sign_name == "right":
                                        if sign_distance <= 20:
                                            direction = 0
                                            previous_signs=[]
                                            turning = True
                                            timer_start = time.time()
                                            print("Tourne à droite")
                                            signs = []

                        elif turning == True:
                            print("En train de tourner...")
                            timer_end = time.time()
                            time_elapsed = timer_end - timer_start
                            print("Timer : " + str(time_elapsed))
                            if time_elapsed >= 1.3:
                                turning = False
                                direction = 90
                                timer_start = 0
                                timer_end = 0
                                time_elapsed = 0
                    elif parked == True:
                        targeted_speed = 0
                        parking_step = 0
                
                    if parking == False:
                        acceleration,objectDetected,initialDistance,real_speed = accelerationFunction(front_distance,targeted_speed,objectDetected,initialDistance)
                    elif parking == True and spot_finded == False:
                        acceleration = 120
                    await websocket.send(str(real_speed))
                    #print("Consigne : " + str(acceleration))
                    servos.servo[1].angle = acceleration
                    servos.servo[2].angle = direction
                    previous_signs = signs
                    #print("La vitesse de la voiture est de " + str(targeted_speed) + "km/h")
                    #print("Acceleration : " + str(acceleration))
                elif stopped == True:
                    timer_end = time.time()
                    time_elapsed = timer_end - timer_start
                    print("Stopped to a stop sign.")
                    if time_elapsed >= 3:
                        print("On repart!")
                        stopped = False
                        targeted_speed = previous_speed
                timer = time.time()

            elif mode == "manual":

                if previous_mode == "autonomous":
                    servos.servo[1].angle = 92
                    servos.servo[2].angle = 90
                    servos.servo[0].angle = 180
                    previous_mode = "manual"
                    stopped = False
                    parking = False
                    parked = False
                    turning = False
                    back_obstacle = False
                    spot_finded = False
                    measuring = False
                    parking_step = 0
                    v_max = 3.6



a = SignDetection()
a.start()
b = AllFunctionnalities()
b.start()
c = Mode()
c.start()

ipServeur = "10.30.50.186"
