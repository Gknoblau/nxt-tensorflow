import nxt.locator
from nxt.sensor import *
from nxt.motor import *
import requests
from threading import Thread
import time

b = nxt.locator.find_one_brick()
motor1 = Motor(b, PORT_B)
motor2 = Motor(b, PORT_C)
sound1 = Sound(b, PORT_1)
sound2 = Sound(b, PORT_4)
ultra1 = Ultrasonic(b, PORT_2)
ultra2 = Ultrasonic(b, PORT_3)

def run_motor1(speed):
    motor1.turn(float(speed) * -1, 270)

def run_motor2(speed):
    motor2.turn(float(speed) * -1, 270)

while True:
    time.sleep(3)
    data0 = str(sound1.get_sample())
    data1 = str(sound2.get_sample())
    data2 = str(ultra1.get_sample())
    data3 = str(ultra2.get_sample())

    all_data = data0 + " " + data1 + " " + data2 + " " + data3
    try:
        result = requests.post('http://192.168.1.79:3000/tf', data = all_data)
        speeds = result.text.split(" ")
        print speeds
        Thread(target=run_motor1, args=(speeds[0], )).start()
        Thread(target=run_motor2, args=(speeds[1], )).start()

    except StandardError as e:
        print "No connection"
        print e
    
    
    
