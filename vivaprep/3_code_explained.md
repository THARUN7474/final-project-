# 💻 Complete Code Explanation — Every Line, Every Function
## UNet → ResNet-34 → ResNet-34+ → Proposed Model

---

## FILE 1: UNet Baseline — Line-by-Line Code Explanation

### STEP 1: Importing Libraries

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Suppress TF debug logs (0=all, 3=errors only)
os.environ['NCCL_DEBUG'] = 'WARN'             # NCCL = GPU communication library, only show warnings

import random                                  # For random number generation
import numpy as np                             # NumPy: numerical array operations (matrices)
import matplotlib.pyplot as plt                # Plotting library for loss curves
import cv2                                     # OpenCV: image processing (flood-fill, morphology)
from glob import glob                          # File path pattern matching (find all *.png files)
from PIL import Image                          # Python Imaging Library: load/resize images

import tensorflow as tf                        # THE deep learning framework

from sklearn.model_selection import train_test_split   # Splits data into train/val sets randomly
from tensorflow.keras.optimizers import Adam           # Adam optimizer for weight updates
from tensorflow.keras.metrics import Recall, Precision # Built-in classification metrics
from tensorflow.keras.layers import (
    Input,              # Defines model input shape
    Conv2D,             # 2D Convolution — the core of CNNs (feature extraction)
    BatchNormalization,  # Normalizes layer outputs — stabilizes training
    Activation,          # Activation function layer
    MaxPool2D,           # Max Pooling — downsamples spatially (takes max value in window)
    Conv2DTranspose,     # Transposed Convolution — upsamples (increases spatial size)
    Concatenate,         # Joins tensors along channel dimension (skip connections)
    LeakyReLU,           # Activation: f(x) = x if x>0, else 0.1*x (prevents dead neurons)
    Add                  # Element-wise addition (for residual connections)
)
from tensorflow.keras.models import Model      # Creates model from input/output layers
```

### STEP 2: Project Constants

```python
H = 256                        # Image height in pixels (model input size)
W = 256                        # Image width in pixels
BATCH_SIZE_PER_REPLICA = 8     # Number of images each GPU processes simultaneously
                               # With 2 GPUs: total batch = 8 × 2 = 16 images
LEARNING_RATE = 1e-4           # 0.0001 — how much weights change per update step
                               # This is a common "safe" starting point for Adam
EPOCHS = 150                   # Maximum number of times to go through entire dataset
                               # (Early stopping usually stops around 29-50)
LR_PATIENCE = 5                # If val_iou doesn't improve for 5 epochs → halve learning rate
ES_PATIENCE = 15               # If val_iou doesn't improve for 15 epochs → stop training
SMOOTH = 1e-6                  # 0.000001 — prevents division by zero in IoU calculation
```

### STEP 3: Data Loading Pipeline

```python
def load_data(base_path):
    """Loads image file paths from the dataset directory structure."""
    
    # DRYADS dataset structure:
    # base_path/
    #   Training/training/
    #     sample_001/images/img.png    ← satellite image tile
    #     sample_001/masks/mask.png    ← binary road mask (white=road, black=background)
    #   Testing/testing/
    #     sample_xxx/images/img.png
    #     sample_xxx/masks/mask.png
    
    train_dir = os.path.join(base_path, "Training", "training")
    test_dir  = os.path.join(base_path, "Testing", "testing")
    
    # glob finds all files matching a pattern
    # "*/images/*.png" means: any_folder/images/any_file.png
    images = sorted(glob(os.path.join(train_dir, "*", "images", "*.png")))
    masks  = sorted(glob(os.path.join(train_dir, "*", "masks", "*.png")))
    # sorted() ensures images[i] corresponds to masks[i] (same sample)
    
    test_images = sorted(glob(os.path.join(test_dir, "*", "images", "*.png")))
    test_masks  = sorted(glob(os.path.join(test_dir, "*", "masks", "*.png")))
    
    # Verify every image has a corresponding mask
    assert len(images) == len(masks)
    
    # Split training data into 80% train, 20% validation
    tx, vx, ty, vy = train_test_split(
        images, masks,        # Split both lists in the same way
        test_size=0.2,        # 20% goes to validation
        random_state=42       # Fixed seed → same split every time (reproducible)
    )
    return (tx, ty), (vx, vy), (test_images, test_masks)


def read_image(path):
    """Load a satellite image and prepare it for the model."""
    img = Image.open(path)           # Open file from disk
    img = img.convert('RGB')         # Ensure 3 channels (Red, Green, Blue)
    img = img.resize((W, H))         # Resize to 256×256 (model expects this size)
    arr = np.array(img, dtype=np.float32)  # Convert to numpy array of floats
    return arr / 255.0               # Normalize pixels from [0-255] to [0.0-1.0]
    # WHY normalize? Neural networks work best with small input values.
    # Pixel value 128 → 0.502. Pixel value 255 → 1.0. Pixel value 0 → 0.0.


def read_mask(path):
    """Load a binary road mask."""
    mask = Image.open(path)          # Open mask file
    mask = mask.convert('L')         # Convert to grayscale (1 channel)
    mask = mask.resize((W, H))       # Resize to 256×256
    arr = np.array(mask, dtype=np.float32) / 255.0  # Normalize: white(255)→1.0, black(0)→0.0
    return np.expand_dims(arr, axis=-1)  # Add channel dimension: [256,256] → [256,256,1]
    # WHY expand_dims? TensorFlow expects shape [H, W, channels]. Mask has 1 channel.


def tf_parse(x, y):
    """Wrap read functions for TensorFlow's data pipeline."""
    def _p(x, y): return read_image(x), read_mask(y)
    # tf.numpy_function bridges NumPy (python) operations into TF's graph
    x, y = tf.numpy_function(_p, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])    # Tell TF the expected output shapes
    y.set_shape([H, W, 1])    # (TF can't infer shapes from numpy_function)
    return x, y
```

### STEP 4: UNet Architecture

```python
def conv_block(x, filters):
    """Two 3×3 convolutions with BatchNorm and LeakyReLU activation.
    
    This is the FUNDAMENTAL building block of UNet.
    
    Corresponds to paper Figure 4: "each module encompassing two layers,
    characterized by 3×3 convolutional operations" + ReLU
    
    Args:
        x: input tensor, e.g. [batch, 256, 256, 3]
        filters: number of output feature maps, e.g. 64
    Returns:
        tensor [batch, 256, 256, filters]
    """
    # FIRST CONVOLUTION
    x = Conv2D(filters, 3, padding='same')(x)
    #   Conv2D(64, 3, padding='same')
    #   - 64 = number of filters (output channels). Each filter detects a different pattern.
    #   - 3 = kernel size (3×3 sliding window)
    #   - padding='same' = pad input with zeros so output has SAME spatial size
    #   
    #   What a 3×3 filter does:
    #   Each filter is a 3×3 grid of learnable weights. It slides across every position
    #   of the input image, computing: output_pixel = Σ(filter_weights × input_patch) + bias
    #   
    #   Example: a horizontal edge detector filter might look like:
    #   [[-1, -1, -1],
    #    [ 0,  0,  0],
    #    [ 1,  1,  1]]
    #   The network LEARNS these filter values during training.
    
    x = BatchNormalization()(x)
    #   Normalizes the output of Conv2D to have mean≈0 and std≈1.
    #   For each feature channel: x_norm = (x - mean(x)) / sqrt(var(x) + eps)
    #   Then applies learned scale/shift: output = gamma * x_norm + beta
    #   
    #   WHY: Without this, different layers have wildly different output ranges.
    #   BatchNorm stabilizes the range, making training faster and more stable.
    
    x = LeakyReLU(negative_slope=0.1)(x)
    #   Activation function: introduces NON-LINEARITY.
    #   LeakyReLU(x) = x           if x > 0
    #                = 0.1 * x     if x ≤ 0
    #
    #   WHY not standard ReLU? ReLU(x<0) = 0, which means:
    #   - Gradient is 0 for negative values → "dead neuron" problem
    #   - Once a neuron goes negative, it can NEVER recover
    #   LeakyReLU keeps a small gradient (0.1) for negative values → neurons stay alive
    
    # SECOND CONVOLUTION (same structure)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    
    return x
    # Output: [batch, same_H, same_W, filters]


def build_unet(input_shape=(256, 256, 3)):
    """
    Complete UNet architecture.
    
    Paper Figure 4 architecture:
    - 4 encoder blocks (down path) with MaxPooling
    - 1 bottleneck block
    - 4 decoder blocks (up path) with Conv2DTranspose + skip connections
    - 1×1 conv for final binary output
    """
    inputs = Input(input_shape)  # [batch, 256, 256, 3] — the satellite image
    
    # ======================== ENCODER (Down Path) ========================
    # Each level: apply conv_block → save for skip connection → MaxPool to downsample
    
    c1 = conv_block(inputs, 64)        # [batch, 256, 256, 64]  — low-level features (edges, colors)
    p1 = MaxPool2D((2, 2))(c1)         # [batch, 128, 128, 64]  — take max value in each 2×2 window
    #   MaxPool2D reduces spatial size by half. It selects the strongest activation
    #   in each 2×2 window, providing translation invariance (small shifts don't matter).
    
    c2 = conv_block(p1, 128)           # [batch, 128, 128, 128] — medium-level features (textures)
    p2 = MaxPool2D((2, 2))(c2)         # [batch, 64,  64,  128]
    
    c3 = conv_block(p2, 256)           # [batch, 64,  64,  256] — mid-level features (patterns)
    p3 = MaxPool2D((2, 2))(c3)         # [batch, 32,  32,  256]
    
    c4 = conv_block(p3, 512)           # [batch, 32,  32,  512] — high-level features (road shapes)
    p4 = MaxPool2D((2, 2))(c4)         # [batch, 16,  16,  512]
    
    # ======================== BOTTLENECK ========================
    bn = conv_block(p4, 1024)          # [batch, 16,  16,  1024] — most compressed representation
    # At this point, the 256×256 image is compressed to 16×16 with 1024 channels.
    # Each "pixel" here represents a 16×16 patch of the original image.
    # This captures the MOST abstract, semantic information about the scene.
    
    # ======================== DECODER (Up Path) ========================
    # Each level: Conv2DTranspose (upsample) → Concatenate with encoder skip → conv_block
    
    d1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn)
    #   [batch, 32, 32, 512]  — learned upsampling (opposite of pooling)
    #   Conv2DTranspose is like Conv2D but in reverse: it INCREASES spatial size.
    #   strides=(2,2) means the output is 2× the input spatial dimensions.
    
    d1 = Concatenate()([d1, c4])       # [batch, 32, 32, 1024]  — SKIP CONNECTION
    #   This is the KEY innovation of UNet.
    #   d1 has semantic information (decoder knows "this region is a road")
    #   c4 has spatial detail (encoder remembers "these exact pixel locations had edges")
    #   Concatenation combines both: semantic context + spatial precision.
    #   Without this, the decoder would guess where road edges are. With it, it KNOWS.
    
    d1 = conv_block(d1, 512)           # [batch, 32, 32, 512]  — process combined features
    
    d2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)  # [batch, 64, 64, 256]
    d2 = Concatenate()([d2, c3])       # [batch, 64, 64, 512]  — skip from c3
    d2 = conv_block(d2, 256)           # [batch, 64, 64, 256]
    
    d3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)  # [batch, 128, 128, 128]
    d3 = Concatenate()([d3, c2])       # [batch, 128, 128, 256]  — skip from c2
    d3 = conv_block(d3, 128)           # [batch, 128, 128, 128]
    
    d4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)   # [batch, 256, 256, 64]
    d4 = Concatenate()([d4, c1])       # [batch, 256, 256, 128]  — skip from c1
    d4 = conv_block(d4, 64)            # [batch, 256, 256, 64]
    
    # ======================== OUTPUT ========================
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
    #   [batch, 256, 256, 1]  — final prediction!
    #   
    #   Conv2D(1, (1,1)):
    #   - 1 output channel (binary: road or not road)
    #   - 1×1 kernel: no spatial mixing, just combines the 64 feature channels into 1 value per pixel
    #   - Equivalent to a per-pixel fully connected layer
    #   
    #   activation='sigmoid':
    #   - Squashes output to [0, 1] range
    #   - Output value = probability that this pixel is a road
    #   - 0.0 = definitely NOT road, 1.0 = definitely road, 0.5 = uncertain
    
    return Model(inputs, outputs)
    # Model() creates a callable model from the input/output tensor graph.
    # This model can be compiled, trained, and used for prediction.
```

### STEP 5: Training Pipeline

```python
# ─── Multi-GPU Setup ───
strategy = tf.distribute.MirroredStrategy()
# MirroredStrategy splits each batch across available GPUs.
# With 2 GPUs and batch=16: GPU0 gets 8 images, GPU1 gets 8 images.
# Both GPUs compute forward/backward pass independently, then
# AVERAGE the gradients and apply the SAME weight update to both copies.
# This makes training ~2× faster.

GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
# = 8 × 2 = 16 images per training step

# ─── Build Data Pipeline ───
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# Creates a dataset from the file path lists. Each element = (image_path, mask_path)

train_dataset = train_dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
# Apply tf_parse to each element: loads image and mask from disk
# AUTOTUNE: TF automatically figures out how many CPU threads to use

train_dataset = train_dataset.shuffle(buffer_size=500)
# Maintains a buffer of 500 samples. When drawing the next sample,
# it randomly picks from the buffer and replaces with a new one.
# This randomizes training order → prevents the model from learning
# batch-order-dependent patterns.

train_dataset = train_dataset.repeat()
# Makes the dataset infinite. Without this, it would stop after
# one pass through all data. .repeat() lets fit() keep drawing batches.

train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
# Groups individual samples into batches of 16.
# Shape goes from [256,256,3] to [16, 256,256,3]

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# While GPU processes current batch, CPU pre-loads the NEXT batch.
# This eliminates waiting time between batches.

# ─── Build & Compile Model ───
with strategy.scope():
    # Everything inside strategy.scope() is replicated across GPUs
    model = build_unet(input_shape=(256, 256, 3))
    
    def iou(y_true, y_pred):
        """Custom IoU metric for monitoring during training."""
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold probabilities → binary
        intersection = tf.reduce_sum(y_true * y_pred)  # Count pixels that are 1 in BOTH
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + SMOOTH) / (union + SMOOTH)  # IoU formula with smoothing
    
    model.compile(
        loss='binary_crossentropy',    # BCE loss for pixel-wise classification
        optimizer=Adam(LEARNING_RATE), # Adam optimizer with lr=0.0001
        metrics=[iou, Precision(), Recall()]  # Track these during training
    )

# ─── Callbacks ───
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_iou',           # Watch validation IoU
        mode='max',                  # We want IoU to be as HIGH as possible
        patience=ES_PATIENCE,        # Stop if no improvement for 15 epochs
        restore_best_weights=True    # After stopping, load weights from best epoch
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_iou',           # Watch validation IoU
        factor=0.5,                  # Multiply lr by 0.5 when plateauing
        patience=LR_PATIENCE,        # Wait 5 epochs before reducing
        min_lr=1e-6                  # Never go below 0.000001
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model_unet_baseline.keras",  # Save file path
        monitor='val_iou',           # Save based on best validation IoU
        mode='max',                  # Higher IoU = better
        save_best_only=True          # Only save if THIS epoch is the best so far
    )
]

# ─── TRAIN! ───
history = model.fit(
    train_dataset,                   # Training data (batched, shuffled, prefetched)
    epochs=EPOCHS,                   # Maximum 150 epochs
    steps_per_epoch=np.ceil(len(train_x)/GLOBAL_BATCH_SIZE).astype(int),
    # How many batches = 1 epoch. ceil(5700/16) = 357 steps
    validation_data=val_dataset,     # Evaluate on validation set after each epoch
    callbacks=callbacks              # EarlyStopping, ReduceLR, ModelCheckpoint
)
# history contains the training logs: loss, val_loss, iou, val_iou per epoch
```

---

## FILE 2: ResNet-34 — What's Different From UNet

### The ONLY New Function: `residual_block`

```python
def residual_block(x, filters):
    """Residual block: 2 convolutions + identity shortcut.
    
    This is the KEY difference from UNet.
    
    Paper Section 2.3.2: "Each module's output was combined with its input 
    through residual connections (aka 'skip connections')."
    
    Architecture:
    
    Input x ─────────────────────────────────── [Conv2D 1×1 → BN] ──→ ADD ──→ ReLU ──→ Output
       │                                                               ↑
       └──→ [Conv2D 3×3 → BN → ReLU → Conv2D 3×3 → BN → ReLU] ──────┘
    """
    
    # SHORTCUT PATH (identity/projection)
    res = Conv2D(filters, (1, 1), padding='same')(x)
    #   1×1 convolution: changes the number of channels WITHOUT changing spatial size.
    #   If input has 3 channels and we need 64 channels, this maps [H,W,3] → [H,W,64].
    #   This is called "projection shortcut" — it projects the input to match output dimensions.
    
    res = BatchNormalization()(res)
    #   Normalize the projected shortcut
    
    # MAIN PATH (same as UNet's conv_block)
    x = conv_block(x, filters)
    #   Two 3×3 convolutions → processes the input to extract features.
    #   This is the "learning" path — it tries to learn useful features.
    
    # RESIDUAL CONNECTION: ADD both paths
    x = Add()([x, res])
    #   Element-wise addition: output[i] = conv_block_output[i] + projected_input[i]
    #   
    #   WHY ADD instead of CONCATENATE?
    #   - Concatenate would DOUBLE the channels (memory and compute intensive)
    #   - Add keeps the same channel count
    #   - Add means the block only needs to learn the RESIDUAL (difference from input)
    #     which is easier than learning the full transformation from scratch
    #   
    #   Mathematical intuition:
    #     Without residual: output = F(x)           ← must learn EVERYTHING
    #     With residual:    output = F(x) + x        ← only learn F(x) = output - x
    #     If the optimal output IS similar to input, F(x) ≈ 0 (easy to learn!)
    
    x = LeakyReLU(negative_slope=0.1)(x)
    #   Final activation after the addition
    
    return x


def build_resnet(input_shape=(256, 256, 3)):
    """ResNet-34: UNet structure but with residual_block instead of conv_block.
    
    Compared to UNet:
    - SAME encoder-decoder structure (4 levels + bottleneck)
    - SAME skip connections (Concatenate)
    - DIFFERENT blocks: residual_block instead of conv_block
    - ADDS: identity shortcuts in every block → prevents vanishing gradients
    """
    inputs = Input(input_shape)
    
    # ENCODER — residual_block instead of conv_block
    c1 = residual_block(inputs, 64);  p1 = MaxPool2D((2, 2))(c1)   # [batch,128,128,64]
    c2 = residual_block(p1, 128);     p2 = MaxPool2D((2, 2))(c2)   # [batch,64,64,128]
    c3 = residual_block(p2, 256);     p3 = MaxPool2D((2, 2))(c3)   # [batch,32,32,256]
    c4 = residual_block(p3, 512);     p4 = MaxPool2D((2, 2))(c4)   # [batch,16,16,512]
    bn = residual_block(p4, 1024)                                    # [batch,16,16,1024]
    
    # DECODER — also uses residual_block
    d1 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(bn)
    d1 = Concatenate()([d1, c4]);     d1 = residual_block(d1, 512)
    d2 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(d1)
    d2 = Concatenate()([d2, c3]);     d2 = residual_block(d2, 256)
    d3 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(d2)
    d3 = Concatenate()([d3, c2]);     d3 = residual_block(d3, 128)
    d4 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(d3)
    d4 = Concatenate()([d4, c1]);     d4 = residual_block(d4, 64)
    
    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(d4)
    return Model(inputs, outputs)

# COMPILATION — same as UNet (BCE loss)
model.compile(loss='binary_crossentropy', optimizer=Adam(LEARNING_RATE), ...)
```

---

## ResNet-34+ — What's Different From ResNet-34

### Differences Are Minimal But Meaningful

```python
# DIFFERENCE 1: Combo Loss (BCE + Dice) instead of pure BCE
def combo_loss(y_true, y_pred):
    """Combined BCE + Dice Loss.
    
    BCE handles per-pixel classification.
    Dice handles overall region overlap.
    Together they give both local and global optimization signals.
    """
    def dice_coef(y_t, y_p):
        # Dice coefficient: 2 * |intersection| / (|pred| + |true|)
        it = tf.reduce_sum(y_t * y_p)      # Intersection: pixels that are 1 in both
        return (2. * it + SMOOTH) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + SMOOTH)
        # Value between 0 (no overlap) and 1 (perfect overlap)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Per-pixel loss
    dice = 1.0 - dice_coef(y_true, y_pred)                     # Region loss (1 - dice → minimize)
    return bce + dice                                            # Combined

# DIFFERENCE 2: Basic augmentation added
def basic_augment(x, y):
    if tf.random.uniform(()) > 0.5:     # 50% chance
        x = tf.image.flip_left_right(x)  # Mirror image horizontally
        y = tf.image.flip_left_right(y)  # Mirror mask the SAME way (keep alignment!)
    if tf.random.uniform(()) > 0.5:     # 50% chance
        x = tf.image.flip_up_down(x)     # Mirror image vertically
        y = tf.image.flip_up_down(y)     # Mirror mask the SAME way
    return x, y

# DIFFERENCE 3: Augmentation is applied to training data
train_dataset = train_dataset.map(basic_augment, ...)

# ARCHITECTURE: Exactly same as ResNet-34 (no structural change)
# The "+" in ResNet-34+ refers to the PAPER's added decoder residual connections,
# which we already have in ResNet-34 (our implementation always uses residual_block in decoder).
```

---

## FILE 3: Proposed Model — Every Novel Addition Explained

### NEW Import: Multiply Layer

```python
from tensorflow.keras.layers import ..., Multiply
# Multiply performs element-wise multiplication: output = x * y
# Used in attention gates to weight encoder features by attention coefficients
```

### NOVEL FUNCTION 1: Attention Gate

```python
def attention_gate(x, g, inter_filters):
    """
    ATTENTION GATE — The architectural novelty of our proposed model.
    
    Purpose: Learns to SUPPRESS irrelevant encoder features before they
    reach the decoder via skip connections.
    
    Reference: Adapted from Oktay et al., "Attention U-Net" (2018), 
    originally for medical image segmentation. Applied here to road extraction.
    
    Args:
        x: Encoder skip features        [batch, H, W, C_encoder]
           Contains ALL features: roads, trees, buildings, soil, clouds...
           High spatial resolution but noisy.
        
        g: Decoder gating signal         [batch, H, W, C_decoder]
           The decoder's "opinion" about what's in each region.
           Lower detail but semantically meaningful ("this area is likely road").
        
        inter_filters: Bottleneck channels (reduces computation)
           Typically half of input channels: 256→128, 128→64, 64→32
    
    Returns:
        Attention-weighted encoder features  [batch, H, W, C_encoder]
        Same shape as x, but road regions amplified, background suppressed.
    
    Mathematical formulation:
        α = σ(W_g · g + W_x · x + b)      ← attention coefficients
        output = α ⊙ x                      ← gated features
        where σ = sigmoid, ⊙ = element-wise multiplication
    """
    
    # STEP 1: Project gating signal to bottleneck dimension
    Wg = Conv2D(inter_filters, (1, 1), padding='same')(g)
    #   [batch, H, W, inter_filters]
    #   1×1 conv transforms decoder signal to bottleneck space.
    #   This is equivalent to multiplying by weight matrix W_g.
    Wg = BatchNormalization()(Wg)
    
    # STEP 2: Project encoder features to same bottleneck dimension
    Wx = Conv2D(inter_filters, (1, 1), padding='same')(x)
    #   [batch, H, W, inter_filters]
    #   Both signals now have the SAME number of channels → can be added
    Wx = BatchNormalization()(Wx)
    
    # STEP 3: Combine signals
    psi = Add()([Wg, Wx])
    #   [batch, H, W, inter_filters]
    #   Element-wise addition: decoder context + encoder detail.
    #   High values where BOTH signals agree "this is important" (road).
    #   Low values where they disagree (decoder says "background", encoder has noise).
    
    psi = LeakyReLU(negative_slope=0.1)(psi)
    #   Non-linear activation — allows complex decision boundaries
    
    # STEP 4: Produce attention map (0-1 per pixel)
    psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(psi)
    #   [batch, H, W, 1]  ← ONE attention weight per spatial position
    #   sigmoid squashes to [0, 1]:
    #     0.0 = "completely suppress this spatial location"
    #     1.0 = "fully pass through this spatial location"
    #     0.5 = "partially pass through"
    
    # STEP 5: Apply attention — multiply encoder features by attention map
    return Multiply()([x, psi])
    #   [batch, H, W, C_encoder]  ← same shape as input x
    #   Each spatial position in x is scaled by its attention coefficient.
    #   
    #   Example: if psi at position (32, 45) = 0.95 (road region):
    #     All 512 channel values at x[batch, 32, 45, :] are multiplied by 0.95
    #     → Almost unchanged (road detail preserved)
    #   
    #   Example: if psi at position (100, 200) = 0.02 (tree canopy):
    #     All 512 channel values at x[batch, 100, 200, :] are multiplied by 0.02
    #     → Nearly zeroed out (tree noise suppressed)
```

### NOVEL FUNCTION 2: Focal Tversky Loss

```python
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    """
    FOCAL TVERSKY LOSS — The loss function novelty of our proposed model.
    
    Purpose: Handles extreme class imbalance (roads = 5-15% of pixels).
    Standard BCE treats all pixels equally → model predicts "not road" for everything.
    FTL specifically penalizes MISSED ROADS more than false alarms.
    
    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with 
    Improved Attention U-Net for Lesion Segmentation" (2019).
    
    Mathematical formulation:
        TI = TP / (TP + α·FN + β·FP)      ← Tversky Index (asymmetric Dice)
        L_FT = (1 - TI)^γ                  ← Focal modulation
    
    Args:
        y_true: Ground truth mask          [batch, 256, 256, 1]  (binary: 0 or 1)
        y_pred: Model prediction           [batch, 256, 256, 1]  (probability: 0.0-1.0)
        alpha:  Weight for False Negatives (missed roads) = 0.7
        beta:   Weight for False Positives (false alarms) = 0.3
        gamma:  Focal parameter (focus on hard pixels) = 0.75
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # COUNT CONFUSION MATRIX ELEMENTS (soft/differentiable versions)
    tp = tf.reduce_sum(y_true * y_pred)
    #   True Positives: pixels where BOTH ground truth AND prediction = 1
    #   Soft version: y_true=1.0, y_pred=0.8 → contribution = 0.8 (partial credit)
    
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    #   False Negatives: pixels where ground truth = 1 BUT prediction = 0
    #   y_true=1.0, y_pred=0.2 → (1-0.2)=0.8 → contribution = 0.8 (high penalty!)
    #   These are MISSED ROADS — the worst error for our application.
    
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    #   False Positives: pixels where ground truth = 0 BUT prediction = 1
    #   y_true=0.0, y_pred=0.9 → contribution = 0.9 (false road detection)
    
    # TVERSKY INDEX (asymmetric version of Dice/F1)
    tversky_index = (tp + SMOOTH) / (tp + alpha * fn + beta * fp + SMOOTH)
    #   numerator: correctly detected road pixels
    #   denominator: correctly detected + weighted errors
    #   
    #   alpha=0.7 on FN: missing a road pixel costs 0.7
    #   beta=0.3 on FP: falsely predicting a road costs only 0.3
    #   Ratio: 0.7/0.3 = 2.33× → missed roads penalized MORE
    #   
    #   This DIRECTLY boosts recall (model tries harder to find roads)
    #   at a controlled cost to precision (some false alarms are acceptable)
    
    # FOCAL MODULATION
    return tf.pow((1 - tversky_index), gamma)
    #   gamma=0.75 makes the loss focus on HARD examples:
    #   - Easy images (TI=0.9): loss = (1-0.9)^0.75 = 0.1^0.75 = 0.178   ← small
    #   - Hard images (TI=0.3): loss = (1-0.3)^0.75 = 0.7^0.75 = 0.757   ← big!
    #   
    #   The optimizer naturally focuses on images where the model struggles most,
    #   which are typically images with faint, partially occluded roads.
```

### NOVEL FUNCTION 3: Connectivity Penalty

```python
def connectivity_penalty(y_true, y_pred):
    """
    CONNECTIVITY PENALTY — Novel topological loss component.
    
    Purpose: Standard pixel-wise losses (BCE, Dice, Tversky) don't care about
    whether predicted roads are CONTINUOUS or FRAGMENTED.
    This penalty compares the EDGE STRUCTURE of prediction vs ground truth.
    
    How it works:
    1. Apply Laplacian filter (edge detector) to both prediction and ground truth
    2. Compute the difference in edge maps
    3. If prediction has EXTRA edges (fragmented roads → many broken segments),
       the penalty is HIGH.
    4. If prediction has SAME edges as ground truth (continuous roads),
       the penalty is LOW.
    
    Mathematical formulation:
        L_conn = mean(|∇²(ŷ) - ∇²(y)|)
        where ∇² is the Laplacian operator (second-order edge detector)
    """
    # Laplacian kernel — a standard 2nd-order edge detection filter
    laplacian_kernel = tf.constant(
        [[0,  1, 0],
         [1, -4, 1],     # Center pixel = -4, neighbors = 1
         [0,  1, 0]],    # Detects pixels that DIFFER from their neighbors
        dtype=tf.float32
    )
    laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])
    #   Reshape to [height=3, width=3, input_channels=1, output_channels=1]
    #   This format is required by tf.nn.conv2d
    
    # Apply Laplacian to prediction → find edges in predicted road mask
    edges_pred = tf.nn.conv2d(y_pred, laplacian_kernel, strides=[1,1,1,1], padding='SAME')
    #   Where prediction transitions from road(1) to background(0):
    #   Laplacian will produce large positive/negative values
    #   Where prediction is smooth (all road or all background): values ≈ 0
    
    # Apply Laplacian to ground truth → find edges in actual road mask
    edges_true = tf.nn.conv2d(y_true, laplacian_kernel, strides=[1,1,1,1], padding='SAME')
    
    # Mean Absolute Error between edge maps
    return tf.reduce_mean(tf.abs(edges_pred - edges_true))
    #   If prediction edges MATCH ground truth edges → difference ≈ 0 → low penalty
    #   If prediction has EXTRA edges (fragmentation) → large values → high penalty
    #   If prediction MISSING edges (over-smoothing) → also detectable → penalty
    #
    #   KEY INSIGHT: This loss is DIFFERENTIABLE (uses conv2d, not connected components),
    #   so gradients can flow through it during backpropagation.
    #   This is why it works as a TRAINING loss, not just a metric.
```

### NOVEL FUNCTION 4: Combined Proposed Loss

```python
def proposed_loss(y_true, y_pred):
    """
    COMBINED LOSS — The complete novel loss function.
    
    L_total = L_FocalTversky + λ × L_Connectivity
    where λ = 0.3
    
    Why λ=0.3?
    - Too high (0.7+): connectivity dominates → model prioritizes edges over pixel accuracy
    - Too low (0.1):   connectivity is ignored → fragmentation not penalized enough  
    - 0.3 = balance: 70% pixel accuracy focus + 30% topology focus
    """
    ftl = focal_tversky_loss(y_true, y_pred)     # Pixel-level: find more roads
    conn = connectivity_penalty(y_true, y_pred)   # Topology-level: keep roads connected
    return ftl + 0.3 * conn                        # Combined
```

### NOVEL: Post-Processing Pipeline (After Prediction)

```python
# This runs AFTER the model makes its prediction — not during training.

# Step 1: Get raw prediction
p = model.predict(np.expand_dims(sample_img, 0))[0]    # [256, 256, 1] probabilities
m = (p.squeeze() > 0.5).astype(np.uint8) * 255          # Threshold → binary (0 or 255)

# Step 2: Flood-Fill to remove border artifacts
# The paper identified border artifacts as a major source of error (Section 3):
# "misclassification of artificial image edges resultant of image processing"
clean = m.copy()
gray = cv2.cvtColor(np.uint8(sample_img*255), cv2.COLOR_RGB2GRAY)
b_mask = (gray == 0).astype(np.uint8)    # Find black border pixels

for s in [(0,0), (255,0), (0,255), (255,255)]:    # Four corners
    f_mask = np.zeros((258, 258), np.uint8)         # FloodFill needs mask 2px larger
    if b_mask[s[1], s[0]] == 1:                     # If corner is black (border)
        cv2.floodFill(clean, f_mask, s, 0)          # Fill connected black region with 0
        # This removes any road prediction that is connected to the image border
        # (which is likely an artifact from image processing, not an actual road)

# Step 3: Morphological Closing to bridge small gaps
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
# Morphological closing = dilation followed by erosion:
#   Dilation: expands white regions by 1 pixel → fills small gaps
#   Erosion:  shrinks white regions by 1 pixel → removes the expansion
#   Net effect: small gaps between road segments are FILLED, 
#              but road width remains approximately the same
```

### NOVEL Evaluation: Connectivity Metric

```python
def connectivity_score(pred_bin, true_bin):
    """
    NOVEL METRIC — Measures topological fidelity of road network.
    
    Uses OpenCV's connected components analysis:
    - Connected component = a group of white pixels that are all touching
    - If ground truth has 5 road segments and prediction has 50 disconnected fragments,
      the score = 5/50 = 0.10 (very fragmented)
    - If prediction has 5 segments too, score = 5/5 = 1.0 (perfect connectivity)
    
    This metric captures something IoU and F1 CANNOT:
    whether the predicted roads form a USABLE network or just scattered dots.
    """
    _, n_pred = cv2.connectedComponents(pred_bin)
    #   Finds connected components in prediction. Returns count.
    #   Example: 50 disconnected road fragments → n_pred = 51 (50 + background)
    
    _, n_true = cv2.connectedComponents(true_bin)
    #   Finds connected components in ground truth. 
    #   Example: 5 continuous road segments → n_true = 6 (5 + background)
    
    return n_true / max(n_pred, 1)
    #   Score = GT_components / Pred_components
    #   Closer to 1.0 = better connectivity (fewer fragments)
    #   << 1.0 = highly fragmented prediction
```

---

## SUMMARY: WHAT MAKES EACH MODEL UNIQUE

| Model | Encoder | Decoder | Skip Connection | Loss | Augmentation | Post-Processing |
|---|---|---|---|---|---|---|
| **UNet** | `conv_block` | `conv_block` | Direct `Concatenate` | BCE | None | None |
| **ResNet-34** | `residual_block` | `residual_block` | Direct `Concatenate` | BCE | None | None |
| **ResNet-34+** | `residual_block` | `residual_block` | Direct `Concatenate` | BCE+Dice | Flip | None |
| **Proposed** | `residual_block` | `residual_block` | **`attention_gate`** → `Concatenate` | **Focal Tversky + Connectivity** | Flip+Brightness+Contrast | **Flood-Fill + Morphology** |

### Lines of Code Changed From ResNet-34 → Proposed

```diff
 # IMPORTS
+from tensorflow.keras.layers import ..., Multiply          # NEW: needed for attention gate

 # ARCHITECTURE
+def attention_gate(x, g, inter_filters):                    # +12 lines: NEW function
+    ...

 # BUILD MODEL — decoder changes:
-d1 = Concatenate()([d1, c4]); ...                          # OLD: direct concatenation
+c4_att = attention_gate(c4, d1, 256)                       # NEW: filter through attention
+d1 = Concatenate()([d1, c4_att]); ...                      # then concatenate

 # LOSS FUNCTION
-model.compile(loss='binary_crossentropy', ...)              # OLD: simple BCE
+def focal_tversky_loss(...):                                # +8 lines: NEW loss
+def connectivity_penalty(...):                              # +5 lines: NEW penalty
+def proposed_loss(...):                                     # +3 lines: NEW combined
+model.compile(loss=proposed_loss, ...)                      # compile with new loss

 # AUGMENTATION
+def heavy_augment(x, y):                                    # +5 lines: brightness/contrast
+    x = tf.image.random_brightness(x, 0.1)
+    x = tf.image.random_contrast(x, 0.9, 1.1)

 # POST-PROCESSING & EVALUATION
+# Flood-fill + morphological closing                        # +15 lines: NEW
+# Connectivity metric evaluation                            # +20 lines: NEW
```
