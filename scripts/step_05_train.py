import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# RTX 5060 Blackwell (sm_120) needs CUDA 12.8 for stable BN kernels;
# system has CUDA 12.4 → hide GPU and train on CPU (~4-5 s/epoch, fine).
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks as cb
import numpy as np, json

LABEL_MAP   = json.load(open("label_map.json"))
NUM_CLASSES = len(LABEL_MAP)  # 8
GRID        = 12
IMG_SIZE    = 96
BATCH_SIZE  = 32

print("Training on CPU (RTX 5060 Blackwell requires CUDA 12.8; system has 12.4)")

# ── Architecture ────────────────────────────────────────────────────────────


def dsc(x, filters, stride=1, name=None):
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same',
                               use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(6., name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                      name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(6., name=f"{name}_r2")(x)
    return x


def build_fomo7(num_classes=NUM_CLASSES):
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    x   = layers.Conv2D(8, 3, strides=2, padding='same',
                        use_bias=False, name='stem')(inp)
    x   = layers.BatchNormalization(name='stem_bn')(x)
    x   = layers.ReLU(6., name='stem_r')(x)
    x   = dsc(x, 16,  stride=1, name='d1')
    x   = dsc(x, 32,  stride=2, name='d2')
    x   = dsc(x, 32,  stride=1, name='d3')
    x   = dsc(x, 128, stride=2, name='d4')
    x   = dsc(x, 128, stride=1, name='d5')
    x   = layers.Conv2D(num_classes, 1, padding='same', name='head')(x)
    out = layers.Softmax(axis=-1, name='output')(x)
    return keras.Model(inp, out, name='WasteNet-FOMO7')


model = build_fomo7()
model.summary()
print(f"Params: {model.count_params():,} | Est. INT8: ~{model.count_params()//1024+20} KB")


# ── Dataset ──────────────────────────────────────────────────────────────────

ELEMENT_SPEC = (
    tf.TensorSpec(shape=(None, 96, 96, 3),   dtype=tf.float32),
    tf.TensorSpec(shape=(None, 12, 12, 8),   dtype=tf.float32),
)
train_ds = tf.data.experimental.load("tfdata/train", ELEMENT_SPEC)
val_ds   = tf.data.experimental.load("tfdata/val",   ELEMENT_SPEC)


# ── Loss & metric ────────────────────────────────────────────────────────────

# BG_WEIGHT=0.001 is the key finding: crushing background weight forces
# the gradient to come 90%+ from foreground cells, which is what matters.
# Uniform FG_WEIGHT=1.0 — per-class boosting was tested and hurt accuracy.
BG_WEIGHT = 0.001
FG_WEIGHT = 1.0


def fomo_loss(y_true, y_pred):
    is_bg    = y_true[..., 0:1]
    weights  = is_bg * BG_WEIGHT + (1.0 - is_bg) * FG_WEIGHT
    eps      = 0.05
    y_smooth = y_true * (1 - eps) + eps / NUM_CLASSES
    cce      = -tf.reduce_sum(y_smooth * tf.math.log(y_pred + 1e-7), axis=-1, keepdims=True)
    return tf.reduce_mean(cce * weights)


class FgAcc(tf.keras.metrics.Metric):
    def __init__(self, name='fg_accuracy', **kw):
        super().__init__(name=name, **kw)
        self.c = self.add_weight(shape=(), initializer='zeros', name='c')
        self.t = self.add_weight(shape=(), initializer='zeros', name='t')

    def update_state(self, y_true, y_pred, sample_weight=None):
        fg    = tf.reduce_max(y_true[..., 1:], axis=-1) > 0
        match = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
        self.c.assign_add(tf.reduce_sum(tf.cast(match & fg, tf.float32)))
        self.t.assign_add(tf.reduce_sum(tf.cast(fg, tf.float32)))

    def result(self):
        return self.c / (self.t + 1e-7)

    def reset_state(self):
        self.c.assign(0.)
        self.t.assign(0.)


# ── Training ─────────────────────────────────────────────────────────────────

EPOCHS  = 250
N_STEPS = EPOCHS * (600 * 7 // BATCH_SIZE)

lr_sched = tf.keras.optimizers.schedules.CosineDecay(3e-4, N_STEPS, alpha=0.01)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_sched),
    loss=fomo_loss,
    metrics=[FgAcc()]
)

os.makedirs("checkpoints", exist_ok=True)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[
        cb.ModelCheckpoint("checkpoints/best.keras", monitor='val_fg_accuracy',
                           save_best_only=True, mode='max'),
        cb.EarlyStopping(monitor='val_fg_accuracy', patience=40,
                         restore_best_weights=True, mode='max'),
        cb.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        cb.CSVLogger('training_log.csv'),
    ]
)

model.save("saved_model")

best_val = max(history.history['val_fg_accuracy'])
print(f"\nBest val_fg_accuracy: {best_val:.4f}")
if best_val < 0.78:
    print("WARNING: val_fg_accuracy below 0.78 target.")
else:
    print("CHECKPOINT PASS: val_fg_accuracy >= 0.78")

print("Done. Check training_log.csv for per-epoch metrics.")
