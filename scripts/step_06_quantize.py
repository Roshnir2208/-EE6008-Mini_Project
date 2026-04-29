import tensorflow as tf, numpy as np, glob, cv2, json

NUM_CLASSES = 8
GRID        = 12
BG_W        = 0.001   # must match training BG_WEIGHT
FG_W        = 1.0


def fomo_loss(y_true, y_pred):
    is_bg = y_true[..., 0:1]
    w     = is_bg * BG_W + (1.0 - is_bg) * FG_W
    ys    = y_true * 0.95 + 0.05 / NUM_CLASSES
    return tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(y_pred + 1e-7), -1, True) * w)


class FgAcc(tf.keras.metrics.Metric):
    def __init__(self, name='fg_accuracy', **kw):
        super().__init__(name=name, **kw)
        self.c = self.add_weight(shape=(), initializer='zeros', name='c')
        self.t = self.add_weight(shape=(), initializer='zeros', name='t')

    def update_state(self, yt, yp, sw=None):
        fg = tf.reduce_max(yt[..., 1:], -1) > 0
        m  = tf.equal(tf.argmax(yt, -1), tf.argmax(yp, -1))
        self.c.assign_add(tf.reduce_sum(tf.cast(m & fg, tf.float32)))
        self.t.assign_add(tf.reduce_sum(tf.cast(fg, tf.float32)))

    def result(self):
        return self.c / (self.t + 1e-7)

    def reset_state(self):
        self.c.assign(0.)
        self.t.assign(0.)


model = tf.keras.models.load_model(
    "checkpoints/best.keras",
    custom_objects={'fomo_loss': fomo_loss, 'FgAcc': FgAcc}
)

ELEMENT_SPEC = (
    tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 12, 12, 8), dtype=tf.float32),
)
val_ds = tf.data.experimental.load("tfdata/val", ELEMENT_SPEC)
_, f32_acc = model.evaluate(val_ds, verbose=0)
print(f"Float32 val fg_accuracy: {f32_acc:.4f}")

# Calibration dataset (300 images from training split)
calib = sorted(glob.glob("augmented/train/*/images/*.jpg"))[:300]


def rep_dataset():
    for p in calib:
        img = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (96, 96))
        yield [np.expand_dims(img.astype(np.float32) / 255.0, 0)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.uint8    # OpenMV camera native
converter.inference_output_type     = tf.float32  # heatmap needs float thresholding

tflite = converter.convert()
path   = "wastenet_fomo7_int8.tflite"
open(path, "wb").write(tflite)
size_kb = len(tflite) / 1024
print(f"INT8 model: {size_kb:.1f} KB -> {path}")
assert size_kb < 200, f"Budget exceeded! {size_kb:.1f} KB"
print(f"CHECKPOINT: model size {size_kb:.1f} KB < 200 KB  PASS")

# Quick INT8 accuracy check on ~128 val samples
interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
s, zp = inp_d['quantization']

cor = tot = 0
for imgs, lbls in val_ds.take(4):   # ~4 batches ≈ 128 samples
    for img, lbl in zip(imgs.numpy(), lbls.numpy()):
        uint8 = np.clip(img / s + zp, 0, 255).astype(np.uint8)
        interp.set_tensor(inp_d['index'], uint8[None])
        interp.invoke()
        pred = interp.get_tensor(out_d['index'])[0]
        fg   = lbl.max(-1) > 0
        cor += ((np.argmax(pred, -1) == np.argmax(lbl, -1)) & fg).sum()
        tot += fg.sum()

int8_acc = cor / max(tot, 1)
drop     = (f32_acc - int8_acc) * 100
print(f"INT8 fg_accuracy sample: {int8_acc:.4f} (drop: {drop:.1f}%)")

if drop > 3.0:
    print("WARNING: >3% drop. Enable QAT fallback (commented block below).")
else:
    print(f"CHECKPOINT: accuracy drop {drop:.1f}% <= 3%  PASS")

# QAT FALLBACK — uncomment only if PTQ drop > 3%
# import tensorflow_model_optimization as tfmot
# qat = tfmot.quantization.keras.quantize_model(model)
# qat.compile(tf.keras.optimizers.Adam(1e-5), fomo_loss, [FgAcc()])
# qat.fit(train_ds, validation_data=val_ds, epochs=10,
#         callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
# converter = tf.lite.TFLiteConverter.from_keras_model(qat)
# ... (same INT8 settings as above)
