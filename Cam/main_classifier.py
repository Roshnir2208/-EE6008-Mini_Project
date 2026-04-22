# OpenMV RT1062 garbage classification runtime
# Uses OpenMV's ml.Model API for TFLite inference.

import sensor
import time
import ml
import uos
import gc

MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.45
USE_CENTER_CROP_160 = False

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
if USE_CENTER_CROP_160:
    sensor.set_windowing((160, 160))
sensor.skip_frames(time=2000)

try:
    load_to_fb = uos.stat(MODEL_PATH)[6] > (gc.mem_free() - (64 * 1024))
    net = ml.Model(MODEL_PATH, load_to_fb=load_to_fb)
except Exception as e:
    raise Exception('Failed to load "trained.tflite": ' + str(e))

try:
    labels = [line.rstrip("\n") for line in open(LABELS_PATH)]
except Exception as e:
    raise Exception('Failed to load "labels.txt": ' + str(e))

print("Model loaded. Classes:", labels)
clock = time.clock()
frame_idx = 0
last_label = "Starting..."
last_score = 0.0

while True:
    clock.tick()
    frame_idx += 1
    img = sensor.snapshot()

    try:
        preds = net.predict([img])[0].flatten().tolist()
        best_idx = 0
        best_score = preds[0]

        for i in range(1, len(preds)):
            if preds[i] > best_score:
                best_score = preds[i]
                best_idx = i

        label = labels[best_idx] if best_idx < len(labels) else "Unknown"
        last_label = label
        last_score = best_score
    except Exception as e:
        if frame_idx % 20 == 0:
            print("Predict error:", e)

    if last_score >= CONFIDENCE_THRESHOLD:
        img.draw_string(4, 4, "%s: %.0f%%" % (last_label, last_score * 100), color=(0, 255, 0), scale=2)
    else:
        img.draw_string(4, 4, "Low confidence", color=(255, 0, 0), scale=2)

    img.draw_string(4, 28, "FPS: %.1f" % clock.fps(), color=(255, 255, 0), scale=1)

    if frame_idx % 10 == 0:
        print("%s = %.3f" % (last_label, last_score))
