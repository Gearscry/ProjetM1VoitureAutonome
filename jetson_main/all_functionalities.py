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

#Fonctions
def accelerationFunction(distance_front,targetedSpeed,objectDetected,initialDistance): #Fonction d'accélération et d'anticollision.
    
    sensorDistance = 12 #Distance qui sépare le devant de la maquette et le capteur.
    criticalDistance = 30*(targetedSpeed/v_max) #Distance critique à laquelle s'arrêter.
    if initialDistance == 0:
        initialDistance = 1
    
    distance = distance_front - sensorDistance - criticalDistance #Distance à parcourir pour s'arrêter.
    print("Distance mesurée devant : " + str(distance))
    if distance > 40 and objectDetected == True: #Si un objet est à une distance suppérieure à 40cm et qu'il était précédemment détecté.
        objectDetected = False #Il n'est plus considéré comme détecté.
        initialDistance = 0 #Remise à 0 de la distance initiale.
    elif distance <= 40 and objectDetected == False: #Si un objet est à une distance inférieure à 40cm et qu'il n'était pas précédemment détecté.
        objectDetected = True #Il est considéré comme détecté.
        initialDistance = distance #Enregistrement de sa distance comme distance initiale.
    
    if objectDetected == True: #Si un objet est détecté.
        print("Obstacle détecté à une distance de " + str(front_distance) + "cm.")
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

    return consigne,objectDetected,initialDistance


#Récupération des informations envoyées par l'IHM
def getInfos():
    time.sleep(0.010)
    if os.stat('data.json').st_size != 0:
        with open('data.json','r') as fp:
            infos = json.load(fp)

    return infos["Control Mode"],infos["Park Mode"]

#Fonction de récupération des panneaux et des informations les concernant.
#def getSigns(websocket):
def getSigns():
    time.sleep(0.010)
    if os.stat('signs.json').st_size != 0:
        with open('signs.json', encoding='utf-8-sig') as data:
            signs = json.load(data)
        if signs["sign"] == "nothing":
            signs = []
        else :
            signs = [[signs["sign"],int(signs["distance"]),bool(signs["detected"])]]
    else :
        signs = []
    return signs

#Fonction de stationnement en créneau à droite de la maquette.
def rightParking():
    return

#Fonction utilitaire pour le stationnement.
def step_timer(duration,parking_step,pause):
    print("Parking step : " + str(parking_step) + ".")
    global measuring
    global timer_start
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
    return parking_step

#Programme principal.
if __name__ == '__main__':
#async def main(websocket):

    #Initialisation de la connexion en Série.
    ser = serial.Serial('/dev/ttyUSB0',9600,timeout=1) #Connexion en série à la carte arduino.
    ser.reset_input_buffer()

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
    mode = "autonomous"
    previous_mode = "autonomous"
    parking_mode = "off"
    front_distance = 400
    right_distance = 400
    back_distance = 400
    left_distance = 400
    spot_finded = False
    objectDetected = False
    initialDistance = 0

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
        mode,parking_mode = getInfos()
        print("Mode : " + str(mode) + ", parking : " + str(parking_mode))

        if parking_mode == "on":
            parking = True
        elif parking_mode == "off":
            parking = False

        if mode == "autonomous":
            if previous_mode == "manual":
                servos.servo[0].angle = 90
                previous_mode = "autonomous"

            line = ser.readline().decode('ISO-8859-1').rstrip()
            #print(line)
            string = line.split(",")
            #print("Distances : " + str(line))
            if len(string) > 1 and len(string[0]) <= 3:
                    front_distance = int(string[0])
                    right_distance = int(string[1])-5
                    back_distance = int(string[2])-11
                    left_distance = int(string[3])-5

            if stopped == False: #Si la voiture n'est pas arrêtée à un stop.
                #print("Voiture pas arrêtée à un stop.") 
                signs = getSigns() #Récupération des panneaux triés dans l'ordre de la distance.

                if parked == False:
                    print("Voiture non garée.")
                    if turning == False:
                        print("Voiture pas en train de tourner")
                        if parking == True:
                            print("Voiture en mode parking")
                            if spot_finded == False:  
                                print("Place toujours pas trouvée.")
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
                                if int(string[2]) <= 20 and back_obstacle == False:
                                    pause = 0
                                    back_obstacle = True
                                    pause_start = time.time()
                                elif int(string[2]) > 20 and back_obstacle == True:
                                    back_obstacle = False
                                    pause_end = time.time()
                                    pause = pause_end - pause_start

                                if back_obstacle == False:
                                    print("Pas d'obstacle derrière")
                                    if parking_step == 0:
                                        #On continue de faire avancer la voiture jusqu'à ce que sont arrière soit à ras de l'obstacle (environs 17 cm)
                                        print("Step 0")
                                        acceleration = 105 + (targeted_speed/v_max) * 30
                                        direction = 180
                                        parking_step = step_timer(0.1,0,pause)
                                    elif parking_step == 1:
                                        #Etape 1 : Placement à 45 degrés
                                        print("Step 1")
                                        direction = 0
                                        acceleration = 80 - (targeted_speed/v_max) * 35    
                                        parking_step = step_timer(1,1,pause)
                                    elif parking_step == 2:
                                        #Etape 2 : Déplacement vers l'arrière roues droites
                                        print("Step 2")
                                        direction = 90
                                        parking_step = step_timer(0.3,2,pause)
                                    elif parking_step == 3:
                                        #Etape 3 : Placement droit
                                        print("Step 3")
                                        direction = 180
                                        acceleration = 80 - (targeted_speed/v_max) * 35    
                                        parking_step = step_timer(1,3,pause)
                                    elif parking_step == 4:
                                        #Etape 4 : Replacement
                                        print("Step 4")
                                        direction = 90
                                        acceleration = 115
                                        if int(string[0]) <= 20 :
                                            acceleration = 92
                                            parked = True
                                else:
                                    print("Obstacle derrière")
                                    acceleration = 92

                        if spot_finded == False:

                            signs = getSigns()

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
                        if time_elapsed >= 1:
                            turning = False
                            direction = 90
                            timer_start = 0
                            timer_end = 0
                            time_elapsed = 0
                elif parked == True:
                    targeted_speed = 0
                
                if parking == False:
                    acceleration,objectDetected,initialDistance = accelerationFunction(front_distance,targeted_speed,objectDetected,initialDistance)
 #               await websocket.send(targeted_speed)
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

#connexionServeur = websockets.serve(main, ipServeur, 8585)
#asyncio.get_event_loop().run_until_complete(connexionServeur)
#asyncio.get_event_loop().run_forever()




    
