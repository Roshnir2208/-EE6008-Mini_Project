# Recyclables Classification - Base Script
# Team: EE6008 Mini Project Group 9

import sensor
import time

# Camera setup
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240 (safe for OpenMV)
sensor.skip_frames(time=2000)

clock = time.clock()

print("System Initialised. Waiting for model integration...")

while True:
    clock.tick()
    
    img = sensor.snapshot()  # Capture image
    
    # Placeholder for ML model
    # prediction = model.classify(img)
    
    print("Capturing image... FPS:", clock.fps())
