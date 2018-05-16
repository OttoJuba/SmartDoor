import RPi.GPIO as GPIO
import time

channel = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.IN)
count = 0

def callback(channel):
	global count
	count += 1
	print "vib count = " + str(count)

GPIO.add_event_detect(channel, GPIO.BOTH, bouncetime=500)
GPIO.add_event_callback(channel, callback)

while True:
	time.sleep(1)
