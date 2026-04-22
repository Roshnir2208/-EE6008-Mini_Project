# OpenMV RT1062 tiny detector runtime
# Model outputs: class probs [7] + bbox [cx, cy, w, h] in normalized coordinates.

import gc
import ml
import sensor
import time
import uos

MODEL_PATH = "detector.tflite"
LABELS_PATH = "labels.txt"
CONF_THRESHOLD = 0.35
MIN_MARGIN = 0.03
SMOOTH = 0.65
STRICT_CONFIDENCE = 0.55
STRICT_MARGIN = 0.08
MIN_BOX_AREA_RATIO = 0.02
MAX_BOX_AREA_RATIO = 0.85
STABLE_FRAMES_REQUIRED = 2


def resolve_existing_path(candidates):
    for p in candidates:
        try:
            uos.stat(p)
            return p
        except Exception:
            pass
    return None


def flatten_to_list(x):
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flatten_to_list(item))
        return out
    if hasattr(x, "flatten"):
        return x.flatten().tolist()
    return [x]


def maybe_dequant_unit(arr):
    # OpenMV may already return float probabilities. If values look int8-like,
    # map roughly back into [0, 1].
    if not arr:
        return arr
    vmin = min(arr)
    vmax = max(arr)
    if vmax > 1.2 or vmin < -0.1:
        out = []
        for v in arr:
            vv = (float(v) + 128.0) / 256.0
            if vv < 0.0:
                vv = 0.0
            if vv > 1.0:
                vv = 1.0
            out.append(vv)
        return out
    return [float(v) for v in arr]


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
# Keep a square FOV so bbox coordinates from the square model map correctly.
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

model_candidates = [
    "detector.tflite",
    "/detector.tflite",
    "/flash/detector.tflite",
    "/sd/detector.tflite",
]
label_candidates = [
    "labels.txt",
    "/labels.txt",
    "/flash/labels.txt",
    "/sd/labels.txt",
]

resolved_model_path = resolve_existing_path(model_candidates)
resolved_labels_path = resolve_existing_path(label_candidates)

if resolved_model_path is None:
    raise Exception('Failed to find detector model. Tried: ' + str(model_candidates))
if resolved_labels_path is None:
    raise Exception('Failed to find labels file. Tried: ' + str(label_candidates))

try:
    load_to_fb = uos.stat(resolved_model_path)[6] > (gc.mem_free() - (64 * 1024))
    net = ml.Model(resolved_model_path, load_to_fb=load_to_fb)
except Exception as e:
    raise Exception('Failed to load detector model from "' + resolved_model_path + '": ' + str(e))

try:
    labels = [line.rstrip("\n") for line in open(resolved_labels_path)]
except Exception as e:
    raise Exception('Failed to load labels from "' + resolved_labels_path + '": ' + str(e))

print("Detector model path:", resolved_model_path)
print("Labels path:", resolved_labels_path)
print("Detector loaded. Labels:", labels)
clock = time.clock()
frame_idx = 0
smoothed_cls = None
smoothed_box = [0.5, 0.5, 0.4, 0.4]
stable_label = "Unknown"
stable_score = 0.0
stable_count = 0
uncertain_frames = 0

while True:
    clock.tick()
    frame_idx += 1
    img = sensor.snapshot()

    try:
        outputs = net.predict([img])
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        cls_raw = None
        box_raw = None

        for out in outputs:
            vals = flatten_to_list(out)
            if len(vals) == len(labels):
                cls_raw = vals
            elif len(vals) == 4:
                box_raw = vals

        if cls_raw is None or box_raw is None:
            if frame_idx % 15 == 0:
                print("Unexpected model outputs")
            img.draw_string(4, 4, "Output parse err", color=(255, 0, 0), scale=1)
            continue

        cls = maybe_dequant_unit(cls_raw)
        box = maybe_dequant_unit(box_raw)

        # Normalize class vector for stable threshold behavior.
        cls_sum = 0.0
        for v in cls:
            cls_sum += v
        if cls_sum > 1e-6:
            cls = [v / cls_sum for v in cls]

        if smoothed_cls is None:
            smoothed_cls = cls
        else:
            smoothed_cls = [SMOOTH * old + (1.0 - SMOOTH) * new for old, new in zip(smoothed_cls, cls)]

        smoothed_box = [SMOOTH * old + (1.0 - SMOOTH) * new for old, new in zip(smoothed_box, box)]

        best_idx = 0
        best_score = smoothed_cls[0]
        second_score = -1.0
        for i in range(1, len(smoothed_cls)):
            if smoothed_cls[i] > best_score:
                second_score = best_score
                best_score = smoothed_cls[i]
                best_idx = i
            elif smoothed_cls[i] > second_score:
                second_score = smoothed_cls[i]

        label = labels[best_idx] if best_idx < len(labels) else "Unknown"

        cx = smoothed_box[0]
        cy = smoothed_box[1]
        bw = smoothed_box[2]
        bh = smoothed_box[3]

        if cx < 0.0:
            cx = 0.0
        if cx > 1.0:
            cx = 1.0
        if cy < 0.0:
            cy = 0.0
        if cy > 1.0:
            cy = 1.0
        if bw < 0.02:
            bw = 0.02
        if bw > 1.0:
            bw = 1.0
        if bh < 0.02:
            bh = 0.02
        if bh > 1.0:
            bh = 1.0

        iw = img.width()
        ih = img.height()
        x1 = int((cx - (bw * 0.5)) * iw)
        y1 = int((cy - (bh * 0.5)) * ih)
        x2 = int((cx + (bw * 0.5)) * iw)
        y2 = int((cy + (bh * 0.5)) * ih)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 >= iw:
            x2 = iw - 1
        if y2 >= ih:
            y2 = ih - 1

        w = x2 - x1
        h = y2 - y1
        if w < 2:
            w = 2
        if h < 2:
            h = 2

        box_area_ratio = (w * h) / float(iw * ih)
        geometry_ok = (box_area_ratio >= MIN_BOX_AREA_RATIO) and (box_area_ratio <= MAX_BOX_AREA_RATIO)

        confident = (best_score >= CONF_THRESHOLD) and ((best_score - second_score) >= MIN_MARGIN)
        strong_confident = (best_score >= STRICT_CONFIDENCE) and ((best_score - second_score) >= STRICT_MARGIN)

        if strong_confident and geometry_ok:
            if label == stable_label:
                stable_count += 1
            else:
                stable_label = label
                stable_score = best_score
                stable_count = 1
            if best_score > stable_score:
                stable_score = best_score
            uncertain_frames = 0
        else:
            uncertain_frames += 1
            if uncertain_frames >= 6 and confident and geometry_ok:
                # Fallback: don't stay uncertain forever when signal is decent.
                stable_label = label
                stable_score = best_score
                stable_count = STABLE_FRAMES_REQUIRED
                uncertain_frames = 0
            else:
                stable_count = 0

        color = (0, 255, 0)
        if not confident:
            color = (255, 165, 0)

        img.draw_rectangle(x1, y1, w, h, color=color, thickness=2)
        if stable_count >= STABLE_FRAMES_REQUIRED:
            img.draw_string(4, 4, "%s: %.0f%%" % (stable_label, stable_score * 100), color=(0, 255, 0), scale=2)
        elif confident and geometry_ok:
            img.draw_string(4, 4, "Locking: %s %.0f%%" % (label, best_score * 100), color=(255, 215, 0), scale=2)
        else:
            img.draw_string(4, 4, "Uncertain: %.0f%%" % (best_score * 100), color=color, scale=2)
        img.draw_string(4, 28, "FPS: %.1f" % clock.fps(), color=(255, 255, 0), scale=1)

        if frame_idx % 10 == 0:
            print("%s %.3f m=%.3f area=%.3f stable=%d" % (label, best_score, (best_score - second_score), box_area_ratio, stable_count))

    except Exception as e:
        if frame_idx % 15 == 0:
            print("Inference error:", e)
        img.draw_string(4, 4, "Inference error", color=(255, 0, 0), scale=1)
