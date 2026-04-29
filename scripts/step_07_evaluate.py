import numpy as np, glob, cv2, json, os
import tensorflow as tf, xml.etree.ElementTree as ET
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

TFLITE    = "wastenet_fomo7_int8.tflite"
LABEL_MAP = json.load(open("label_map.json"))
NUM_CLS   = len(LABEL_MAP)  # 8
CLS_NAMES = [LABEL_MAP[str(i)] for i in range(NUM_CLS)]  # index 0 = background
CONF_THR  = 0.45
GRID      = 12
IMG_SIZE  = 96
CELL      = IMG_SIZE // GRID

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
s, zp = inp_d['quantization']


def grid_from_xml(xp, w=96, h=96):
    grid = np.zeros((GRID, GRID), dtype=np.int32)
    if not os.path.exists(xp):
        return grid
    name_to_idx = {v: int(k) for k, v in LABEL_MAP.items()}
    for obj in ET.parse(xp).findall("object"):
        nm  = obj.find("name").text.strip().lower()
        idx = name_to_idx.get(nm, 0)
        bb  = obj.find("bndbox")
        cx  = (float(bb.find("xmin").text) + float(bb.find("xmax").text)) / 2
        cy  = (float(bb.find("ymin").text) + float(bb.find("ymax").text)) / 2
        col = min(int(cx / (w / GRID)), GRID - 1)
        row = min(int(cy / (h / GRID)), GRID - 1)
        grid[row, col] = idx
    return grid


y_true, y_pred = [], []
for cls_idx in range(1, NUM_CLS):
    cls = LABEL_MAP[str(cls_idx)]
    for ip in glob.glob(f"augmented/test/{cls}/images/*.jpg"):
        stem = os.path.splitext(os.path.basename(ip))[0]
        img  = cv2.resize(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        u8   = np.clip(img.astype(np.float32) / 255. / s + zp, 0, 255).astype(np.uint8)
        interp.set_tensor(inp_d['index'], u8[None])
        interp.invoke()
        hm = interp.get_tensor(out_d['index'])[0]  # (12, 12, 8)
        gt = grid_from_xml(f"augmented/test/{cls}/Annotations/{stem}.xml")
        for r in range(GRID):
            for c in range(GRID):
                pc = int(np.argmax(hm[r, c]))
                tc = int(gt[r, c])
                if hm[r, c, pc] < CONF_THR:
                    pc = 0
                if tc > 0 or pc > 0:
                    y_true.append(tc)
                    y_pred.append(pc)

print(classification_report(y_true, y_pred, target_names=CLS_NAMES, zero_division=0))

cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLS)))
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(NUM_CLS))
ax.set_xticklabels(CLS_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(NUM_CLS))
ax.set_yticklabels(CLS_NAMES, fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("WasteNet-FOMO7 Confusion Matrix (test split)")
for i in range(NUM_CLS):
    for j in range(NUM_CLS):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                color='white' if cm[i, j] > cm.max() // 2 else 'black')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
