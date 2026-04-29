# main.py -- copy to OpenMV SD card alongside wastenet_fomo7_int8.tflite
import sensor, tf, time

LABELS = ["background",
          "cardboard", "garbage", "glass", "metal", "paper", "plastic", "trash"]
# NOTE: order must match label_map.json (sorted alphabetically, index 1..7)

COLORS = [(0, 0, 0),        # 0: background (unused)
          (255, 200, 60),   # 1: cardboard  - yellow
          (160, 80, 255),   # 2: garbage    - purple
          (80, 200, 255),   # 3: glass      - cyan
          (80, 80, 255),    # 4: metal      - blue
          (255, 255, 255),  # 5: paper      - white
          (255, 130, 60),   # 6: plastic    - orange
          (255, 60, 60)]    # 7: trash      - red

CONF_THR = 0.50
GRID     = 12
CELL_W   = 8   # 96 / 12
CELL_H   = 8

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.B96X96)
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.skip_frames(time=2000)

net   = tf.load("/sd/wastenet_fomo7_int8.tflite", load_to_fb=True)
clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    try:
        outputs  = net.run([img])
        heatmap  = outputs[0]   # flat list: 12*12*8 = 1152 floats
        NUM_CLS  = 8
        for row in range(GRID):
            for col in range(GRID):
                base     = (row * GRID + col) * NUM_CLS
                probs    = heatmap[base : base + NUM_CLS]
                best_cls = max(range(NUM_CLS), key=lambda i: probs[i])
                conf     = probs[best_cls]
                if conf >= CONF_THR and best_cls > 0:
                    cx    = (col * CELL_W) + CELL_W // 2
                    cy    = (row * CELL_H) + CELL_H // 2
                    col_c = COLORS[best_cls]
                    lbl   = LABELS[best_cls]
                    img.draw_cross(cx, cy, color=col_c, size=5, thickness=2)
                    img.draw_rectangle(cx - 1, cy - 13, len(lbl) * 6 + 28, 12,
                                       color=col_c, fill=True)
                    img.draw_string(cx + 1, cy - 13,
                                    "%s %d%%" % (lbl, int(conf * 100)),
                                    color=(0, 0, 0), scale=1)
        img.draw_string(2, 2, "FPS:%.1f" % clock.fps(), color=(255, 255, 255), scale=1)
    except Exception as e:
        img.draw_string(2, 2, str(e)[:20], color=(255, 0, 0), scale=1)
