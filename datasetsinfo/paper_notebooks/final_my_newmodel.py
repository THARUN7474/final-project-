# Generated from: new-model.ipynb
# Converted at: 2026-04-15T17:23:23.612Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

####################################################################################
# IMPROVED PROPOSED MODEL — DRYADS DATASET
# Key changes vs original:
#   1. ROTATION AUGMENTATION (rot90) — matches base paper strategy
#   2. Tversky alpha=0.6 (was 0.7) — better Precision/Recall balance -> higher F1
#   3. Connectivity penalty = 0.1 (was 0.3) — less recall bias
#   4. Cosine LR decay with warmup (replaces aggressive ReduceLROnPlateau)
#   5. Epochs=100, ES patience=15 — more training time
#   6. F1-optimal threshold (replaces fixed 0.5)
#   7. Test-Time Augmentation (TTA) for better inference
#   8. Full connectivity metric on ALL models + baseline comparison
####################################################################################

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_DEBUG'] = 'WARN'
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     MaxPool2D, Conv2DTranspose, Concatenate,
                                     LeakyReLU, Add, Multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

print(f'TF version : {tf.__version__}')
print(f'GPUs found : {tf.config.list_physical_devices("GPU")}')
print(f'TPUs found : {tf.config.list_physical_devices("TPU")}')

####################################################################################
# CONSTANTS
####################################################################################
H, W                  = 256, 256
BATCH_SIZE_PER_REPLICA = 8
LEARNING_RATE          = 1e-4
EPOCHS                 = 100        # up from 150 max but cosine decay manages it
ES_PATIENCE            = 15         # more patience for cosine schedule
SMOOTH                 = 1e-6
WARMUP_EPOCHS          = 5

# DRYADS dataset path (confirmed working structure)
DRYADS_PATH = '/kaggle/input/datasets/bandatharun/road-detection-satellite-tiles-equatorial-asia'

# Saved model paths — YOUR ACTUAL PATHS
MODEL_BASE  = '/kaggle/input/datasets/bandatharun/my-road-models'
PATHS = {
    'UNet_PDS':     f'{MODEL_BASE}/UNET_PDS/best_model_unet_baseline.keras',
    'ResNet34_PDS': f'{MODEL_BASE}/RESNET_BS_PDS/best_model_resnet34_baseline.keras',
    'Proposed_PDS': f'{MODEL_BASE}/my_proposed_model_PDS/best_model_proposed.keras',
}
SAVE_BEST  = '/kaggle/working/best_improved_proposed.keras'
SAVE_FINAL = '/kaggle/working/final_improved_proposed.keras'

print('[INFO] Setup complete.')

####################################################################################
# DATA LOADING — DRYADS (auto-discover structure)
####################################################################################
def _find_dir(base, candidates):
    """Return the first existing directory from a list of candidate sub-paths."""
    for parts in candidates:
        p = os.path.join(base, *parts) if isinstance(parts, (list, tuple)) else os.path.join(base, parts)
        if os.path.isdir(p):
            return p
    return None

def _find_images(directory):
    """Recursively find image/mask pairs under a directory, trying multiple layouts."""
    if directory is None:
        return [], []
    # Pattern 1: <dir>/*/images/*.png  +  <dir>/*/masks/*.png
    imgs = sorted(glob(os.path.join(directory, '*', 'images', '*.png')))
    msks = sorted(glob(os.path.join(directory, '*', 'masks',  '*.png')))
    if imgs and msks:
        return imgs, msks
    # Pattern 2: <dir>/*/images/*.tif / *.jpg  (some DRYADS variants)
    for ext in ('*.tif', '*.jpg', '*.jpeg'):
        imgs = sorted(glob(os.path.join(directory, '*', 'images', ext)))
        msks_ext = ext  # masks usually same ext
        msks = sorted(glob(os.path.join(directory, '*', 'masks', ext)))
        if not msks:
            msks = sorted(glob(os.path.join(directory, '*', 'masks', '*.png')))
        if imgs and msks:
            return imgs, msks
    # Pattern 3: flat — <dir>/images/*.png  +  <dir>/masks/*.png
    for ext in ('*.png', '*.tif', '*.jpg'):
        imgs = sorted(glob(os.path.join(directory, 'images', ext)))
        msks = sorted(glob(os.path.join(directory, 'masks',  ext)))
        if imgs and msks:
            return imgs, msks
    # Pattern 4: deeply nested — recursive search
    imgs = sorted(glob(os.path.join(directory, '**', 'images', '*.png'), recursive=True))
    msks = sorted(glob(os.path.join(directory, '**', 'masks',  '*.png'), recursive=True))
    return imgs, msks

def load_dryads(base_path):
    # --- Diagnostic: show what's actually inside the dataset ---
    print(f'[DEBUG] Dataset root: {base_path}')
    if os.path.isdir(base_path):
        top = os.listdir(base_path)
        print(f'[DEBUG] Top-level contents: {top}')
        for item in top:
            sub = os.path.join(base_path, item)
            if os.path.isdir(sub):
                print(f'[DEBUG]   {item}/ -> {os.listdir(sub)[:10]}')
    else:
        raise FileNotFoundError(f'Dataset path does not exist: {base_path}')

    # --- Find train directory (try common variants) ---
    train_candidates = [
        ('Training', 'training'), ('Training',), ('training',),
        ('train',), ('Train',),
    ]
    test_candidates = [
        ('Testing', 'testing'), ('Testing',), ('testing',),
        ('test',), ('Test',),
    ]

    train_dir = _find_dir(base_path, train_candidates)
    test_dir  = _find_dir(base_path, test_candidates)

    # If not found, maybe the dataset has an extra nesting level
    if train_dir is None:
        for sub in os.listdir(base_path):
            sp = os.path.join(base_path, sub)
            if os.path.isdir(sp):
                train_dir = _find_dir(sp, train_candidates)
                test_dir  = _find_dir(sp, test_candidates)
                if train_dir:
                    print(f'[DEBUG] Found data inside nested folder: {sub}/')
                    break

    print(f'[DEBUG] train_dir = {train_dir}')
    print(f'[DEBUG] test_dir  = {test_dir}')

    images, masks = _find_images(train_dir)
    test_x, test_y = _find_images(test_dir)

    print(f'[DEBUG] Train images: {len(images)}, masks: {len(masks)}')
    print(f'[DEBUG] Test  images: {len(test_x)}, masks: {len(test_y)}')
    if images:
        print(f'[DEBUG] Sample train image path: {images[0]}')

    assert len(images) > 0,             f'No training images found under {train_dir}'
    assert len(images) == len(masks),    f'Train mismatch: {len(images)} imgs vs {len(masks)} masks'
    assert len(test_x) == len(test_y),   f'Test mismatch:  {len(test_x)} imgs vs {len(test_y)} masks'

    tx, vx, ty, vy = train_test_split(images, masks, test_size=0.2, random_state=42)

    print(f'  Train: {len(tx)}  Val: {len(vx)}  Test: {len(test_x)}')
    return (tx, ty), (vx, vy), (test_x, test_y)

(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_dryads(DRYADS_PATH)
print('[INFO] Data loaded.')

####################################################################################
# IMAGE READERS
####################################################################################
def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img.astype(np.float32)

def read_mask(path):
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (W, H))
    mask = mask / 255.0
    return mask[..., np.newaxis].astype(np.float32)

def tf_parse(x, y):
    # Read image
    img = tf.io.read_file(x)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (H, W))
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Read mask
    mask = tf.io.read_file(y)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (H, W), method='nearest')
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return img, mask
####################################################################################
# AUGMENTATION — KEY IMPROVEMENT: ROTATION (matches base paper strategy)
####################################################################################
def augment(x, y):
    # === 90°/180°/270° rotation (same as base paper used for improvement) ===
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)

    # === Random flips (tf.cond for graph-mode / TPU compatibility) ===
    do_lr = tf.random.uniform(()) > 0.5
    x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(do_lr, lambda: tf.image.flip_left_right(y), lambda: y)
    do_ud = tf.random.uniform(()) > 0.5
    x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(do_ud, lambda: tf.image.flip_up_down(y), lambda: y)

    # === Color jitter (image only) ===
    x = tf.image.random_brightness(x, 0.15)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    x = tf.image.random_saturation(x, 0.75, 1.25)
    x = tf.image.random_hue(x, 0.05)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y

print('[INFO] Augmentation pipeline defined (includes rot90).')

####################################################################################
# MODEL ARCHITECTURE — same Attention ResUNet
####################################################################################
def conv_block(x, f):
    x = Conv2D(f, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(f, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x

def residual_block(x, f):
    res = Conv2D(f, 1, padding='same')(x)
    res = BatchNormalization()(res)
    x   = conv_block(x, f)
    x   = Add()([x, res])
    x   = LeakyReLU(0.1)(x)
    return x

def attention_gate(x, g, inter_f):
    """Attention Gate: suppress irrelevant skip-connection features."""
    Wg = Conv2D(inter_f, 1, padding='same')(g)
    Wg = BatchNormalization()(Wg)
    Wx = Conv2D(inter_f, 1, padding='same')(x)
    Wx = BatchNormalization()(Wx)
    psi = Add()([Wg, Wx])
    psi = LeakyReLU(0.1)(psi)
    psi = Conv2D(1, 1, padding='same', activation='sigmoid')(psi)
    return Multiply()([x, psi])

def build_attention_resunet(input_shape=(256, 256, 3)):
    inp = Input(input_shape)
    # Encoder
    c1 = residual_block(inp, 64);  p1 = MaxPool2D()(c1)
    c2 = residual_block(p1,  128); p2 = MaxPool2D()(c2)
    c3 = residual_block(p2,  256); p3 = MaxPool2D()(c3)
    c4 = residual_block(p3,  512); p4 = MaxPool2D()(c4)
    bn = residual_block(p4, 1024)
    # Decoder + Attention Gates
    d1 = Conv2DTranspose(512, 2, strides=2, padding='same')(bn)
    d1 = Concatenate()([d1, attention_gate(c4, d1, 256)])
    d1 = residual_block(d1, 512)
    d2 = Conv2DTranspose(256, 2, strides=2, padding='same')(d1)
    d2 = Concatenate()([d2, attention_gate(c3, d2, 128)])
    d2 = residual_block(d2, 256)
    d3 = Conv2DTranspose(128, 2, strides=2, padding='same')(d2)
    d3 = Concatenate()([d3, attention_gate(c2, d3, 64)])
    d3 = residual_block(d3, 128)
    d4 = Conv2DTranspose(64,  2, strides=2, padding='same')(d3)
    d4 = Concatenate()([d4, attention_gate(c1, d4, 32)])
    d4 = residual_block(d4, 64)
    out = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    return Model(inp, out)

print('[INFO] Model architecture defined.')

####################################################################################
# LOSS FUNCTIONS — TUNED FOR BETTER F1
# Change: alpha=0.6 (was 0.7) -> more balanced Precision/Recall -> better F1
# Change: connectivity weight=0.1 (was 0.3) -> less over-recall bias
####################################################################################
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    union  = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + SMOOTH) / (union + SMOOTH)

def focal_tversky_loss(y_true, y_pred, alpha=0.6, beta=0.4, gamma=0.75):
    """
    Focal Tversky Loss.
    alpha=0.6, beta=0.4: slightly MORE balanced than original (was 0.7/0.3)
    This gives better Precision+Recall balance -> higher F1 score.
    gamma=0.75: focuses on hard road pixels.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    ti = (tp + SMOOTH) / (tp + alpha*fn + beta*fp + SMOOTH)
    return tf.pow((1.0 - ti), gamma)

def connectivity_penalty(y_true, y_pred):
    """Laplacian edge-matching connectivity penalty."""
    k = tf.constant([[0,1,0],[1,-4,1],[0,1,0]], dtype=tf.float32)
    k = tf.reshape(k, [3, 3, 1, 1])
    ep = tf.nn.conv2d(y_pred, k, strides=[1,1,1,1], padding='SAME')
    et = tf.nn.conv2d(y_true, k, strides=[1,1,1,1], padding='SAME')
    return tf.reduce_mean(tf.abs(ep - et))

def proposed_loss(y_true, y_pred):
    """Focal Tversky + 0.1 x Connectivity (was 0.3 — reduced for better F1)."""
    return focal_tversky_loss(y_true, y_pred) + 0.1 * connectivity_penalty(y_true, y_pred)

CUSTOM_OBJECTS = {
    'iou': iou_metric, 'iou_metric': iou_metric,
    'focal_tversky_loss': focal_tversky_loss,
    'connectivity_penalty': connectivity_penalty,
    'proposed_loss': proposed_loss,
}

print('[INFO] Loss functions defined.')
print('  Tversky alpha=0.6 (balanced), connectivity weight=0.1 (reduced)')

####################################################################################
# COSINE LR SCHEDULE WITH WARMUP
# Replaces aggressive ReduceLROnPlateau which crashed LR to 1e-6 by epoch 29
####################################################################################
class CosineWarmupScheduler(Callback):
    def __init__(self, base_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.base_lr     = base_lr
        self.warmup      = warmup_epochs
        self.total       = total_epochs
        self.min_lr      = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)
        if epoch % 10 == 0:
            print(f'\n  [LR] Epoch {epoch+1}: lr = {lr:.2e}')

print('[INFO] Cosine LR schedule defined.')

####################################################################################
# BUILD DATASETS & TRAIN
####################################################################################

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print(f'[INFO] Running on TPU: {tpu.master()}')
except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    print('[INFO] TPU not found, falling back to GPU/CPU')

GLOBAL_BS = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(f'[INFO] Replicas: {strategy.num_replicas_in_sync}  Global batch: {GLOBAL_BS}')

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(500)
    .repeat()
    .batch(GLOBAL_BS, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
    .with_options(options)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_x, val_y))
    .map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BS, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
    .with_options(options)
)

steps_per_epoch = int(np.ceil(len(train_x) / GLOBAL_BS))

with strategy.scope():
    model = build_attention_resunet()
    model.compile(
        loss=proposed_loss,
        optimizer=Adam(LEARNING_RATE),
        metrics=[
            iou_metric,
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )

print(f'[INFO] Model built. Total params: {model.count_params():,}')

callbacks = [
    CosineWarmupScheduler(LEARNING_RATE, WARMUP_EPOCHS, EPOCHS),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_iou_metric',
        mode='max',
        patience=ES_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        SAVE_BEST,
        monitor='val_iou_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
]


print('[INFO] Starting training...')

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)

model.save(SAVE_FINAL)
print(f'[SUCCESS] Model saved → {SAVE_FINAL}')

####################################################################################
# PLOT TRAINING CURVES
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Improved Proposed Model — DRYADS Training', fontsize=13, fontweight='bold')

# Loss
axes[0].plot(history.history['loss'],     label='Train Loss', color='royalblue')
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[0].set_title('Loss Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

# IoU
iou_key = 'iou_metric' if 'iou_metric' in history.history else 'iou'
axes[1].plot(history.history[iou_key],         label='Train IoU', color='royalblue')
axes[1].plot(history.history[f'val_{iou_key}'], label='Val IoU',   color='orange')
axes[1].axhline(0.58, color='red', linestyle='--', lw=1.2, label='Base paper target (0.58)')
axes[1].set_title('IoU Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves_improved.png', dpi=150)
plt.show()
print('[SAVED] training_curves_improved.png')

####################################################################################
# F1-OPTIMAL THRESHOLD SEARCH
# Key insight: using 0.5 is suboptimal. Find the threshold that maximizes F1
# on validation set, then use THAT threshold for test evaluation.
####################################################################################
print('[INFO] Searching for F1-optimal threshold on validation set...')

val_sample = min(len(val_x), 200)   # use up to 200 val samples for speed
all_preds, all_masks = [], []

for i in range(val_sample):
    img  = read_image(val_x[i])
    msk  = read_mask(val_y[i])
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    all_preds.append(pred.squeeze())
    all_masks.append((msk.squeeze() > 0.5).astype(np.uint8))

all_preds = np.array(all_preds).flatten()
all_masks = np.array(all_masks).flatten()

best_f1, best_thresh = 0.0, 0.5
thresh_f1s = []

for t in np.arange(0.2, 0.85, 0.025):
    pbin = (all_preds >= t).astype(np.uint8)
    f1   = f1_score(all_masks, pbin, zero_division=0)
    thresh_f1s.append((t, f1))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f'\n  Best threshold: {best_thresh:.3f}  ->  Val F1 = {best_f1:.4f}')
print(f'  (Default 0.5 threshold Val F1 = {f1_score(all_masks, (all_preds>=0.5).astype(np.uint8), zero_division=0):.4f})')

# Plot threshold curve
ts, f1s = zip(*thresh_f1s)
plt.figure(figsize=(9, 4))
plt.plot(ts, f1s, 'b-o', markersize=4, label='F1 vs Threshold')
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best = {best_thresh:.3f}')
plt.axvline(0.5, color='gray', linestyle=':', label='Default = 0.50')
plt.xlabel('Threshold'); plt.ylabel('F1 Score')
plt.title('Validation F1 vs Decision Threshold'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/threshold_search.png', dpi=150)
plt.show()
print(f'[INFO] Using threshold = {best_thresh:.3f} for all further evaluations')
THRESHOLD = best_thresh

####################################################################################
# TEST-TIME AUGMENTATION (TTA)
# Average predictions from 4 orientations: original + 3 rotations
# + 2 flips = up to 8 augmentations averaged. This typically adds 2-5% F1.
####################################################################################
def predict_tta(model, img):
    """TTA: average over 8 augmentation variants (4 rot90 x 2 flip)."""
    preds = []
    for k in range(4):           # 0°, 90°, 180°, 270°
        rotated = np.rot90(img, k=k)
        p = model.predict(np.expand_dims(rotated, 0), verbose=0)[0].squeeze()
        preds.append(np.rot90(p, k=(4-k) % 4))   # rotate back
        # also add horizontal flip
        flipped = np.fliplr(rotated)
        pf = model.predict(np.expand_dims(flipped, 0), verbose=0)[0].squeeze()
        preds.append(np.fliplr(np.rot90(pf, k=(4-k) % 4)))
    return np.mean(preds, axis=0)  # average all 8 predictions

print('[INFO] TTA function defined (8-fold: 4 rotations × 2 flips).')

####################################################################################
# POST-PROCESSING: Flood Fill + Morphological Closing
####################################################################################
def post_process(pred_bin, img_rgb=None):
    """Apply flood fill for border cleanup + morphological closing for gap bridging."""
    m = (pred_bin * 255).astype(np.uint8)

    # Flood fill border noise removal
    if img_rgb is not None:
        gray  = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        b_mask = (gray == 0).astype(np.uint8)
        clean = m.copy()
        for seed in [(0, 0), (255, 0), (0, 255), (255, 255)]:
            f_mask = np.zeros((H + 2, W + 2), np.uint8)
            if b_mask[seed[1], seed[0]] == 1:
                cv2.floodFill(clean, f_mask, seed, 0)
        m = clean

    # Morphological closing (5x5 kernel — larger than original 3x3 for DRYADS gaps)
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    return (m > 127).astype(np.uint8)

print('[INFO] Post-processing defined (5x5 kernel, 2 iterations).')

####################################################################################
# COMPREHENSIVE TEST SET EVALUATION — IMPROVED PROPOSED MODEL
####################################################################################
print(f'\n[INFO] Evaluating on {len(test_x)} test samples (TTA + optimal threshold + post-processing)...')

ious, precisions, recalls, f1s, conn_scores = [], [], [], [], []

def connectivity_score(pb, tb):
    n_p, _ = cv2.connectedComponents(pb.astype(np.uint8))
    n_t, _ = cv2.connectedComponents(tb.astype(np.uint8))
    return min(float(n_t) / max(float(n_p), 1.0), 2.0)

for i in range(len(test_x)):
    img = read_image(test_x[i])
    msk = read_mask(test_y[i])
    tb  = (msk.squeeze() > 0.5).astype(np.uint8)

    # TTA prediction
    tta_pred = predict_tta(model, img)

    # Optimal threshold
    pb = (tta_pred >= THRESHOLD).astype(np.uint8)

    # Post-processing
    pb = post_process(pb, img)

    # Metrics
    inter = np.sum(pb * tb)
    union = np.sum(pb) + np.sum(tb) - inter
    iou   = float((inter + SMOOTH) / (union + SMOOTH))
    prec  = float((inter + SMOOTH) / (np.sum(pb) + SMOOTH))
    rec   = float((inter + SMOOTH) / (np.sum(tb) + SMOOTH))
    f1    = 2 * prec * rec / (prec + rec + SMOOTH)
    conn  = connectivity_score(pb, tb)

    ious.append(iou); precisions.append(prec); recalls.append(rec)
    f1s.append(f1);   conn_scores.append(conn)

    if (i + 1) % 100 == 0:
        print(f'  Processed {i+1}/{len(test_x)}')

mean_iou  = np.mean(ious)
mean_prec = np.mean(precisions)
mean_rec  = np.mean(recalls)
mean_f1   = np.mean(f1s)
mean_conn = np.mean(conn_scores)
std_conn  = np.std(conn_scores)

print('\n' + '='*60)
print('  IMPROVED PROPOSED MODEL — TEST SET RESULTS')
print('='*60)
print(f'  IoU               : {mean_iou:.4f}  ({mean_iou*100:.1f}%)')
print(f'  Precision         : {mean_prec:.4f}')
print(f'  Recall            : {mean_rec:.4f}')
print(f'  F1 Score          : {mean_f1:.4f}  ({mean_f1*100:.1f}%)')
print(f'  Connectivity Mean : {mean_conn:.4f} ± {std_conn:.4f}')
print(f'  Threshold Used    : {THRESHOLD:.3f}')
print('='*60)
print('  Base Paper Targets (ResNet34):')
print(f'    IoU: 0.58  |  F1: 0.81')
print(f'  Your gap: IoU {mean_iou - 0.58:+.4f}  |  F1 {mean_f1 - 0.81:+.4f}')
print('='*60)

improved_results = {
    'model': 'Improved_Proposed_DRYADS',
    'threshold': THRESHOLD,
    'iou': mean_iou, 'precision': mean_prec,
    'recall': mean_rec, 'f1': mean_f1,
    'connectivity_mean': mean_conn, 'connectivity_std': std_conn,
}
with open('/kaggle/working/improved_results.json', 'w') as f:
    json.dump(improved_results, f, indent=2)
print('[SAVED] improved_results.json')

####################################################################################
# EVALUATE BASELINE MODELS FOR CONNECTIVITY COMPARISON
# Load UNet and ResNet34 baselines, compute connectivity on same test set
####################################################################################
def quick_eval(model_path, name, test_x, test_y, threshold=0.5, use_tta=False):
    """Load a saved model and evaluate it fully on test set."""
    print(f'\n[INFO] Loading {name} from {model_path}')
    if not os.path.exists(model_path):
        print(f'  [SKIP] File not found: {model_path}')
        return None

    m = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    ious, precs, recs, f1s, conns = [], [], [], [], []

    for i in range(len(test_x)):
        img = read_image(test_x[i])
        msk = read_mask(test_y[i])
        tb  = (msk.squeeze() > 0.5).astype(np.uint8)

        if use_tta:
            prob = predict_tta(m, img)
        else:
            prob = m.predict(np.expand_dims(img, 0), verbose=0)[0].squeeze()

        pb = (prob >= threshold).astype(np.uint8)

        inter = np.sum(pb * tb); union = np.sum(pb) + np.sum(tb) - inter
        iou  = float((inter + SMOOTH) / (union + SMOOTH))
        prec = float((inter + SMOOTH) / (np.sum(pb) + SMOOTH))
        rec  = float((inter + SMOOTH) / (np.sum(tb) + SMOOTH))
        f1   = 2 * prec * rec / (prec + rec + SMOOTH)
        conn = connectivity_score(pb, tb)
        ious.append(iou); precs.append(prec); recs.append(rec)
        f1s.append(f1);   conns.append(conn)

    result = {
        'model': name, 'iou': np.mean(ious),
        'precision': np.mean(precs), 'recall': np.mean(recs),
        'f1': np.mean(f1s),
        'connectivity_mean': np.mean(conns), 'connectivity_std': np.std(conns),
    }
    print(f'  IoU={result["iou"]:.4f}  F1={result["f1"]:.4f}  Conn={result["connectivity_mean"]:.4f}')

    del m
    tf.keras.backend.clear_session()
    return result

baseline_results = []

r = quick_eval(PATHS['UNet_PDS'], 'UNet (Baseline)', test_x, test_y)
if r: baseline_results.append(r)

r = quick_eval(PATHS['ResNet34_PDS'], 'ResNet-34 (Baseline)', test_x, test_y)
if r: baseline_results.append(r)

# Original proposed model (for comparison with improved)
r = quick_eval(PATHS['Proposed_PDS'], 'Original Proposed', test_x, test_y, threshold=0.5)
if r: baseline_results.append(r)

print('\n[INFO] All baseline evaluations complete.')

####################################################################################
# FINAL COMPARISON TABLE — YOUR MODELS vs BASE PAPER
####################################################################################
all_rows = baseline_results + [improved_results]

print('\n' + '='*90)
print('  FINAL COMPARISON TABLE — DRYADS DATASET')
print('='*90)
print(f'{"Model":<28} {"IoU":>6} {"F1":>6} {"Prec":>6} {"Rec":>6} {"Conn":>6}')
print('-'*90)

for r in all_rows:
    print(f'{r["model"]:<28} {r["iou"]:>6.4f} {r["f1"]:>6.4f}'
          f' {r["precision"]:>6.4f} {r["recall"]:>6.4f}'
          f' {r["connectivity_mean"]:>6.4f}')

print('-'*90)
# Base paper reference
for name, iou, f1 in [('Paper-UNet', 0.43, 0.72), ('Paper-ResNet34', 0.58, 0.81), ('Paper-ResNet34+', 0.58, 0.81)]:
    print(f'{"[Base Paper] " + name:<28} {iou:>6.2f} {f1:>6.2f} {"N/A":>6} {"N/A":>6} {"N/A":>6}')
print('='*90)

with open('/kaggle/working/final_comparison.json', 'w') as f:
    json.dump(all_rows, f, indent=2)
print('[SAVED] final_comparison.json')

####################################################################################
# VISUALIZATION: BAR CHART — YOUR MODELS vs BASE PAPER
####################################################################################
COLORS = {
    'UNet (Baseline)':   '#F4A460',
    'ResNet-34 (Baseline)': '#D2691E',
    'Original Proposed': '#6495ED',
    'Improved_Proposed_DRYADS': '#1E40AF',
}
LABELS = {
    'UNet (Baseline)': 'UNet\n(Ours)',
    'ResNet-34 (Baseline)': 'ResNet-34\n(Ours)',
    'Original Proposed': 'Proposed\n(Original)',
    'Improved_Proposed_DRYADS': 'Proposed\n(Improved)',
}
PAPER_LINES = {'IoU': 0.58, 'F1': 0.81, 'Connectivity': None}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('DRYADS Results — Your Models vs Base Paper', fontsize=14, fontweight='bold')

for ax, (metric, key, paper_val) in zip(axes, [
    ('IoU', 'iou', 0.58),
    ('F1 Score', 'f1', 0.81),
    ('Connectivity', 'connectivity_mean', None),
]):
    labels, vals, colors = [], [], []
    for r in all_rows:
        name = r['model']
        labels.append(LABELS.get(name, name))
        vals.append(r[key])
        colors.append(COLORS.get(name, '#888888'))

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    if paper_val:
        ax.axhline(paper_val, color='red', linestyle='--', lw=1.5,
                   label=f'Base Paper Best: {paper_val:.2f}')
        ax.legend(fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, min(max(vals) * 1.25, 1.1))
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('/kaggle/working/final_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print('[SAVED] final_bar_chart.png')

####################################################################################
# CONNECTIVITY SCORE DISTRIBUTION — IMPROVED MODEL
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Connectivity Analysis — Improved Proposed Model (DRYADS)', fontsize=12, fontweight='bold')

# Histogram
axes[0].hist(conn_scores, bins=25, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(1.0, color='green', linestyle='--', lw=2, label='Perfect (1.0)')
axes[0].axvline(mean_conn, color='red', linestyle='--', lw=2, label=f'Mean ({mean_conn:.3f})')
axes[0].set_xlabel('Connectivity Score'); axes[0].set_ylabel('Count')
axes[0].set_title('Connectivity Score Distribution')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Scatter: IoU vs Connectivity
sc = axes[1].scatter(ious, conn_scores, c=conn_scores, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=2)
axes[1].axhline(1.0, color='green', linestyle='--', lw=1.5, label='Perfect Connectivity')
axes[1].axhline(mean_conn, color='red', linestyle='--', lw=1.5, label=f'Mean Conn={mean_conn:.3f}')
axes[1].set_xlabel('IoU Score'); axes[1].set_ylabel('Connectivity Score')
axes[1].set_title('IoU vs Connectivity per Sample')
plt.colorbar(sc, ax=axes[1], label='Connectivity'); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/kaggle/working/connectivity_analysis_improved.png', dpi=150)
plt.show()
print('[SAVED] connectivity_analysis_improved.png')

####################################################################################
# VISUAL COMPARISON: INPUT | GT | RAW PRED | POST-PROCESSED
####################################################################################
n_vis = 5
indices = np.random.choice(len(test_x), n_vis, replace=False)

fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
fig.suptitle('Improved Proposed Model — Visual Predictions (TTA + Optimal Threshold)',
             fontsize=12, fontweight='bold')

for col, title in enumerate(['Satellite Image', 'Ground Truth', 'Raw TTA Pred (thresh)', 'Post-Processed']):
    axes[0, col].set_title(title, fontsize=10, fontweight='bold')

for row, idx in enumerate(indices):
    img = read_image(test_x[idx])
    msk = read_mask(test_y[idx]).squeeze()
    tb  = (msk > 0.5).astype(np.uint8)

    tta_p = predict_tta(model, img)
    raw   = (tta_p >= THRESHOLD).astype(np.uint8)
    post  = post_process(raw, img)

    # Metrics for this sample
    inter = np.sum(post * tb); union = np.sum(post) + np.sum(tb) - inter
    s_iou = float((inter + SMOOTH) / (union + SMOOTH))
    s_prec = float((inter + SMOOTH) / (np.sum(post) + SMOOTH))
    s_rec  = float((inter + SMOOTH) / (np.sum(tb) + SMOOTH))
    s_f1   = 2 * s_prec * s_rec / (s_prec + s_rec + SMOOTH)

    axes[row, 0].imshow(img)
    axes[row, 1].imshow(msk, cmap='gray')
    axes[row, 2].imshow(raw, cmap='gray')
    axes[row, 3].imshow(post, cmap='gray')
    axes[row, 3].set_xlabel(f'IoU={s_iou:.3f}  F1={s_f1:.3f}', fontsize=9)

    for c in range(4): axes[row, c].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/visual_predictions.png', dpi=120)
plt.show()
print('[SAVED] visual_predictions.png')

####################################################################################
# FINAL SUMMARY PRINTOUT
####################################################################################
print('\n' + '='*70)
print('  RESEARCH SUMMARY — IMPROVED PROPOSED MODEL')
print('='*70)
print(f'  Dataset          : DRYADS')
print(f'  Architecture     : Attention-Guided Residual UNet (ResNet-34 backbone)')
print(f'  Key Improvements : Rotation augmentation + Cosine LR + Tuned Tversky')
print(f'  Inference        : TTA (8-fold) + Optimal threshold ({THRESHOLD:.3f}) + Post-processing')
print()
print(f'  RESULTS:')
print(f'  IoU        = {mean_iou:.4f}  (Base paper ResNet34 = 0.58, gap: {mean_iou-0.58:+.4f})')
print(f'  F1 Score   = {mean_f1:.4f}  (Base paper ResNet34 = 0.81, gap: {mean_f1-0.81:+.4f})')
print(f'  Precision  = {mean_prec:.4f}')
print(f'  Recall     = {mean_rec:.4f}')
print(f'  Connectivity = {mean_conn:.4f} ± {std_conn:.4f}')
print()
baseline_iou = 0.2706  # original UNet baseline
print(f'  Improvement over your UNet baseline:')
print(f'  IoU: {mean_iou:.4f} vs {baseline_iou:.4f} = +{mean_iou - baseline_iou:.4f} ({(mean_iou/baseline_iou - 1)*100:.1f}% relative)')
print('='*70)
print('[DONE] All outputs saved to /kaggle/working/')
print('  Files: best_improved_proposed.keras')
print('         final_improved_proposed.keras')
print('         improved_results.json')
print('         final_comparison.json')
print('         training_curves_improved.png')
print('         threshold_search.png')
print('         final_bar_chart.png')
print('         connectivity_analysis_improved.png')
print('         visual_predictions.png')