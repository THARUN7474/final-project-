# 🏗️ Architecture Deep-Dive — All 4 Models Compared
## Paper Architecture vs Our Code | Every Layer, Every Difference

---

## 1. COMPLETE ARCHITECTURE COMPARISON TABLE

| Component | UNet | ResNet-34 | ResNet-34+ | Proposed (Attn-ResUNet) |
|---|---|---|---|---|
| **Encoder Backbone** | Standard Conv blocks | Residual Conv blocks | Residual Conv blocks | Residual Conv blocks |
| **Encoder Blocks** | 4 levels + bottleneck | 4 levels + bottleneck | 4 levels + bottleneck | 4 levels + bottleneck |
| **Encoder Block Type** | 2× Conv2D + BN + ReLU | 2× Conv2D + BN + ReLU + **Skip/Add** | 2× Conv2D + BN + ReLU + **Skip/Add** | 2× Conv2D + BN + ReLU + **Skip/Add** |
| **Filter Progression** | 64→128→256→512→1024 | 64→128→256→512→1024 | 64→128→256→512→1024 | 64→128→256→512→1024 |
| **Downsampling** | MaxPool2D(2,2) | MaxPool2D(2,2) | MaxPool2D(2,2) | MaxPool2D(2,2) |
| **Decoder Blocks** | 4 levels | 4 levels | 4 levels | 4 levels |
| **Decoder Block Type** | Conv blocks (no residual) | Residual blocks | Residual blocks | Residual blocks |
| **Upsampling** | Conv2DTranspose(2,2) | Conv2DTranspose(2,2) | Conv2DTranspose(2,2) | Conv2DTranspose(2,2) |
| **Skip Connections** | ✅ Direct Concatenate | ✅ Direct Concatenate | ✅ Direct Concatenate | ✅ **Attention-Gated** then Concatenate |
| **Residual (Identity) Shortcuts** | ❌ None | ✅ In encoder only | ✅ In encoder + **decoder** | ✅ In encoder + decoder |
| **Attention Gates** | ❌ None | ❌ None | ❌ None | ✅ **4 attention gates** |
| **Loss Function** | BCE | BCE | BCE + Dice (combo) | **Focal Tversky + 0.3× Connectivity** |
| **Data Augmentation** | None | None | Flip (H/V) | Flip + Brightness + Contrast |
| **Post-Processing** | None | None | None | **Flood-Fill + Morphological Closing** |
| **Connectivity Metric** | ❌ | ❌ | ❌ | ✅ Novel metric |
| **Total Parameters** | ~31.0M | ~32.4M | ~32.4M | ~33.1M |

---

## 2. WHAT EACH MODEL ADDS OVER THE PREVIOUS ONE

```
UNet (Baseline)
  │
  │  + Add residual/identity shortcuts inside encoder blocks
  ▼
ResNet-34
  │
  │  + Add residual shortcuts in decoder blocks too
  │  + Use combo loss (BCE + Dice) instead of pure BCE
  │  + Add basic augmentation (flip)
  ▼
ResNet-34+
  │
  │  + Add Attention Gates on ALL skip connections
  │  + Replace loss with Focal Tversky + Connectivity
  │  + Add heavy augmentation (flip + brightness + contrast)
  │  + Add post-processing (flood-fill + morphology)
  │  + Add connectivity metric evaluation
  ▼
Proposed (Attention-Guided ResUNet)
```

---

## 3. THE KEY ARCHITECTURAL DIFFERENCE — BLOCK BY BLOCK

### UNet: conv_block (No Residual)

```
Input x ──→ [Conv2D 3×3] ──→ [BatchNorm] ──→ [LeakyReLU] ──→ [Conv2D 3×3] ──→ [BatchNorm] ──→ [LeakyReLU] ──→ Output
```

The input goes through two convolutions and comes out. **No shortcut.** If the block can't learn useful features, the original input information is lost.

### ResNet-34: residual_block (With Identity Shortcut)

```
Input x ──→ [Conv2D 3×3] ──→ [BatchNorm] ──→ [LeakyReLU] ──→ [Conv2D 3×3] ──→ [BatchNorm] ──→ [LeakyReLU] ──→ ADD ──→ [LeakyReLU] ──→ Output
                                                                                                                  ↑
Input x ──→ [Conv2D 1×1] ──→ [BatchNorm] ───────────────────────────────────────────────────────────────────────────┘
            (identity/projection shortcut)
```

The input is processed through two convolutions AND simultaneously projected through a 1×1 conv. Both paths are **added together**. This means: even if the conv block learns nothing useful, the original information passes directly through via the shortcut. This solves the **vanishing gradient problem** — gradients can flow directly through the skip path during backpropagation.

### ResNet-34+: Same encoder + residual decoder + combo loss

The decoder blocks also use `residual_block` instead of `conv_block`. The paper describes this as:
> *"residual connections were added between each of the max pooling layers and the up-sampling layers to preserve the data between the encoding and decoding layers"*

Additionally, ResNet-34+ uses a **combo loss** (BCE + Dice combined):
```python
def combo_loss(y_true, y_pred):
    bce = binary_crossentropy(y_true, y_pred)       # Standard pixel-wise loss
    dice = 1.0 - dice_coefficient(y_true, y_pred)   # Region-overlap loss
    return bce + dice                                 # Combined: local + global awareness
```

### Proposed: Attention Gates + Novel Loss (The Key Novelty)

```
                                    ENCODER                    DECODER
                              ┌──────────────┐          ┌──────────────┐
    Input ──→ ResidualBlock ──│ c1 (64 ch)   │─── ATTENTION GATE ──→ │ Concat + ResBlock │──→ Output
              MaxPool         │              │    ↑     ↑  (c1_att)  │ d4 (64 ch)       │
              ResidualBlock ──│ c2 (128 ch)  │─── ATTENTION GATE ──→ │ Concat + ResBlock │
              MaxPool         │              │    ↑     ↑  (c2_att)  │ d3 (128 ch)      │
              ResidualBlock ──│ c3 (256 ch)  │─── ATTENTION GATE ──→ │ Concat + ResBlock │
              MaxPool         │              │    ↑     ↑  (c3_att)  │ d2 (256 ch)      │
              ResidualBlock ──│ c4 (512 ch)  │─── ATTENTION GATE ──→ │ Concat + ResBlock │
              MaxPool         │              │         ↑  (c4_att)  │ d1 (512 ch)      │
              ResidualBlock ──│ bn (1024 ch) │─────────────────────→ │ Conv2DTranspose   │
                              └──────────────┘                       └──────────────────┘
```

The attention gate is the critical difference. Instead of blindly passing encoder features to the decoder (as UNet/ResNet do), the attention gate **learns which spatial regions are important** based on what the decoder has already figured out.

---

## 4. ATTENTION GATE — Detailed Mechanism

### What It Does

The attention gate takes TWO inputs:
1. **x** = encoder features (high spatial detail, but noisy — contains roads + trees + everything)
2. **g** = decoder gating signal (lower spatial detail, but semantically rich — "knows" where roads likely are)

It outputs: **x_weighted** = same spatial detail as x, but road regions amplified, non-road regions suppressed.

### The Math

```
Step 1: Project both inputs to same channel dimension
  Wg = Conv2D(inter_filters, 1×1)(g)     →  Transform decoder signal
  Wx = Conv2D(inter_filters, 1×1)(x)     →  Transform encoder features

Step 2: Combine and produce attention coefficients
  psi = LeakyReLU(Wg + Wx)               →  Element-wise addition + activation
  psi = Conv2D(1, 1×1, sigmoid)(psi)      →  Squash to [0,1] per pixel

Step 3: Apply attention to encoder features
  output = x * psi                         →  Element-wise multiplication
```

### Visual Example — What Attention Does to a Road Image

```
ENCODER FEATURES (x):          ATTENTION MAP (psi):         GATED OUTPUT (x × psi):
┌──────────────────┐            ┌──────────────────┐         ┌──────────────────┐
│ Trees: 0.8       │            │ Trees: 0.05      │         │ Trees: 0.04      │  ← SUPPRESSED
│ Road:  0.6       │     ×      │ Road:  0.95      │    =    │ Road:  0.57      │  ← PRESERVED
│ Soil:  0.7       │            │ Soil:  0.10      │         │ Soil:  0.07      │  ← SUPPRESSED
│ Building:  0.9   │            │ Building: 0.02   │         │ Building: 0.018  │  ← SUPPRESSED
└──────────────────┘            └──────────────────┘         └──────────────────┘

Result: Only road features survive. Trees, soil, buildings are suppressed.
```

### In Our Code — The Attention Gate Function

```python
def attention_gate(x, g, inter_filters):
    # x: encoder skip features     [batch, H, W, channels_x]     e.g. [16, 64, 64, 256]
    # g: decoder gating signal      [batch, H, W, channels_g]     e.g. [16, 64, 64, 256]
    # inter_filters: bottleneck     e.g. 128
    
    Wg = Conv2D(inter_filters, (1,1), padding='same')(g)    # [16, 64, 64, 128]  ← compress decoder signal
    Wg = BatchNormalization()(Wg)                            # Normalize for stable training
    
    Wx = Conv2D(inter_filters, (1,1), padding='same')(x)    # [16, 64, 64, 128]  ← compress encoder features
    Wx = BatchNormalization()(Wx)                            # Normalize
    
    psi = Add()([Wg, Wx])                                    # [16, 64, 64, 128]  ← combine signals
    psi = LeakyReLU(negative_slope=0.1)(psi)                 # Non-linear activation
    
    psi = Conv2D(1, (1,1), padding='same', activation='sigmoid')(psi)  
    # [16, 64, 64, 1]  ← attention map: one weight per spatial position
    # sigmoid → all values between 0 (suppress) and 1 (keep)
    
    return Multiply()([x, psi])                               # [16, 64, 64, channels_x]
    # Element-wise: x × psi → road features kept, background suppressed
```

### Where Attention Gates Are Placed (4 Total)

```python
# In build_resnet() of our Proposed Model:

# Level 1 (512 channels): Decoder d1 gets attention-filtered c4
d1 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(bn)
c4_att = attention_gate(c4, d1, 256)         # ← ATTENTION GATE: filter c4 using d1
d1 = Concatenate()([d1, c4_att])             # Concatenate filtered features
d1 = residual_block(d1, 512)

# Level 2 (256 channels): Decoder d2 gets attention-filtered c3
d2 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(d1)
c3_att = attention_gate(c3, d2, 128)         # ← ATTENTION GATE
d2 = Concatenate()([d2, c3_att])
d2 = residual_block(d2, 256)

# Level 3 (128 channels): Decoder d3 gets attention-filtered c2
d3 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(d2)
c2_att = attention_gate(c2, d3, 64)          # ← ATTENTION GATE
d3 = Concatenate()([d3, c2_att])
d3 = residual_block(d3, 128)

# Level 4 (64 channels): Decoder d4 gets attention-filtered c1
d4 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(d3)
c1_att = attention_gate(c1, d4, 32)          # ← ATTENTION GATE
d4 = Concatenate()([d4, c1_att])
d4 = residual_block(d4, 64)
```

---

## 5. DATA FLOW THROUGH EACH MODEL — TENSOR SIZES AT EACH STEP

### UNet: Complete Forward Pass

```
INPUT: [batch=16, 256, 256, 3]  (RGB satellite image)

ENCODER:
  c1 = conv_block(input, 64)     →  [16, 256, 256, 64]    # 64 feature maps learned
  p1 = MaxPool2D(2,2)(c1)        →  [16, 128, 128, 64]    # Spatial size halved
  
  c2 = conv_block(p1, 128)       →  [16, 128, 128, 128]
  p2 = MaxPool2D(2,2)(c2)        →  [16, 64,  64,  128]
  
  c3 = conv_block(p2, 256)       →  [16, 64,  64,  256]
  p3 = MaxPool2D(2,2)(c3)        →  [16, 32,  32,  256]
  
  c4 = conv_block(p3, 512)       →  [16, 32,  32,  512]
  p4 = MaxPool2D(2,2)(c4)        →  [16, 16,  16,  512]

BOTTLENECK:
  bn = conv_block(p4, 1024)      →  [16, 16,  16,  1024]  # Maximum compression

DECODER:
  u1 = Conv2DTranspose(512)      →  [16, 32,  32,  512]   # Upsample
  d1 = Concat([u1, c4])          →  [16, 32,  32,  1024]  # Skip connection! 512+512
  d1 = conv_block(d1, 512)       →  [16, 32,  32,  512]
  
  u2 = Conv2DTranspose(256)      →  [16, 64,  64,  256]
  d2 = Concat([u2, c3])          →  [16, 64,  64,  512]   # 256+256
  d2 = conv_block(d2, 256)       →  [16, 64,  64,  256]
  
  u3 = Conv2DTranspose(128)      →  [16, 128, 128, 128]
  d3 = Concat([u3, c2])          →  [16, 128, 128, 256]   # 128+128
  d3 = conv_block(d3, 128)       →  [16, 128, 128, 128]
  
  u4 = Conv2DTranspose(64)       →  [16, 256, 256, 64]
  d4 = Concat([u4, c1])          →  [16, 256, 256, 128]   # 64+64
  d4 = conv_block(d4, 64)        →  [16, 256, 256, 64]

OUTPUT:
  out = Conv2D(1, sigmoid)       →  [16, 256, 256, 1]     # Probability map
```

### ResNet-34: Same as UNet BUT residual_block replaces conv_block

```
ONLY DIFFERENCE FROM UNET:

conv_block:
  x = Conv(x) → BN → ReLU → Conv → BN → ReLU → return x

residual_block:
  res = Conv1x1(input) → BN          # Shortcut: project input to match output channels
  x = Conv(input) → BN → ReLU → Conv → BN → ReLU
  x = Add([x, res])                   # ADD shortcut to processed output
  x = ReLU(x)                         # Final activation
  return x
```

### ResNet-34+: Same as ResNet-34 BUT decoder also uses residual blocks + combo loss

```
UNet decoder:         conv_block(concat_features, filters)
ResNet-34 decoder:    residual_block(concat_features, filters)    ← SAME
ResNet-34+ decoder:   residual_block(concat_features, filters)    ← SAME architecture
                      + combo_loss (BCE + Dice)                    ← DIFFERENT loss
                      + basic_augment (flip H/V)                   ← ADDS augmentation
```

### Proposed Model: ResNet-34 encoder + Attention gates + Novel loss

```
DECODER (the only structural difference):

  u1 = Conv2DTranspose(512)(bn)        →  [16, 32, 32, 512]
  c4_att = attention_gate(c4, u1, 256) →  [16, 32, 32, 512]    # ← NEW: filter c4
  d1 = Concat([u1, c4_att])           →  [16, 32, 32, 1024]
  d1 = residual_block(d1, 512)        →  [16, 32, 32, 512]
  
  (same pattern for d2, d3, d4 with attention gates)

Plus:
  loss = focal_tversky_loss + 0.3 × connectivity_penalty     ← NEW loss
  augment = flip + brightness + contrast                       ← HEAVIER augmentation
  post-process = flood_fill + morphological_closing            ← NEW post-processing
```

---

## 6. WHY EACH IMPROVEMENT MATTERS — THE RESEARCH LOGIC

### UNet → ResNet-34: Why Add Residual Connections?

**Problem:** In deep networks, gradients become very small as they flow backward through many layers (vanishing gradient problem). With plain conv blocks, the early encoder layers barely learn.

**Solution:** The residual shortcut `x = Add([conv_out, identity(x)])` ensures gradients can flow directly through the Add() operation. Even if `conv_out` has zero gradient, the `identity(x)` path carries gradient backward.

**Mathematical Proof:**
```
Without residual: gradient = ∂loss/∂x_layer1 = ∂loss/∂x_L × Π(∂x_i/∂x_{i-1})
                  → product of many small numbers → vanishes to ~0

With residual:    x_out = F(x) + x
                  ∂x_out/∂x = ∂F(x)/∂x + 1    ← the "+1" prevents vanishing!
                  Gradient is ALWAYS at least 1, no matter how deep
```

### ResNet-34 → ResNet-34+: Why Add Decoder Residual + Combo Loss?

**Problem:** The decoder reconstructs the road mask from compressed features. Without residual connections in the decoder, fine spatial details are lost during upsampling.

**Solution:** Residual decoder preserves details. Combo loss (BCE + Dice) adds a global region-aware term so the model considers overall road segment overlap, not just individual pixels.

### ResNet-34+ → Proposed: Why Add Attention Gates?

**Problem with Skip Connections:**
Skip connections pass ALL encoder features to the decoder — including tree textures, soil patterns, cloud shadows, and building edges that look like roads.

**Solution:**
Attention gates learn to **filter** these features BEFORE concatenation:
- Road-like features → attention coefficient ≈ 1 → passed through
- Background features → attention coefficient ≈ 0 → suppressed

**Impact on Results:**

| What Changed | Effect on Metrics |
|---|---|
| Attention gates | ↑ Recall (find more roads, fewer false negatives) |
| Focal Tversky Loss | ↑ Recall (penalize missed roads more) + ↑ F1 |
| Connectivity Penalty | ↑ Connectivity (fewer broken road segments) |
| Heavy Augmentation | ↓ Overfitting gap (model sees varied inputs) |
| Post-Processing | ↑ Visual quality (remove noise, bridge gaps) |

---

## 7. PAPER'S ARCHITECTURE DESCRIPTIONS vs OUR CODE

### Paper's UNet (Section 2.3.1, Figure 4) vs Our Code

| Paper Description | Our Implementation |
|---|---|
| *"3×3 convolutional operations devoid of padding"* | `Conv2D(filters, 3, padding='same')` — We use 'same' padding to preserve spatial dimensions |
| *"ReLU activation"* | `LeakyReLU(negative_slope=0.1)` — We use Leaky ReLU for better gradient flow |
| *"2×2 max-pooling layer, stride of 2"* | `MaxPool2D((2, 2))` — Exactly as described |
| *"Feature maps: 64, 128, 512, 1024"* | Our filters: 64, 128, 256, 512, 1024 — We include 256 |
| *"2×2 transposed convolution"* | `Conv2DTranspose(filters, (2,2), strides=(2,2))` — Exact match |
| *"1×1 convolutional operation on final layer"* | `Conv2D(1, (1,1), activation='sigmoid')` — Exact match |
| *"concatenated"* skip connections | `Concatenate()([decoder, encoder])` — Exact match |
| BatchNormalization | Added by us — not mentioned in paper but standard practice |

### Paper's ResNet-34 (Section 2.3.2, Figure 5) vs Our Code

| Paper Description | Our Implementation |
|---|---|
| *"16 modules, each having 2 convolutional layers with 3×3 kernel and ReLU"* | We use 4 residual blocks in encoder (simplified from 16) due to compute constraints |
| *"residual connections"* | `Add()([conv_output, projected_input])` — Exact implementation |
| *"max pooling operation with stride 2"* | `MaxPool2D((2,2))` — Exact match |
| *"3 consecutive up-sampling layers with stride of 2"* | We use 4 `Conv2DTranspose` layers (matching our 4 encoder levels) |
| *"2×2 transpose convolution operation"* | `Conv2DTranspose(filters, (2,2), strides=(2,2))` — Exact match |

### Paper's ResNet-34+ (Section 2.3.3, Figure 6) vs Our Code

| Paper Description | Our Implementation |
|---|---|
| *"residual connections between max pooling and up-sampling layers"* | Decoder uses `residual_block` (same as encoder block type) |
| *"Layers were joined using concatenation operation"* | `Concatenate()` — Exact match |
| *"fewer up-sampling operations to preserve data"* | We maintain 4 decoder levels to match encoder |

### What the Paper Had That We DON'T Have

| Paper Feature | Why We Differ |
|---|---|
| 2-stage training | Single-stage: fair comparison across models |
| Up to 1,500 total epochs | 50-150 epochs: Kaggle compute limits |
| Image rotation augmentation | We use flips + photometric augmentation instead |
| Possibly SGD optimizer | We use Adam (adaptive, converges faster with fewer epochs) |
