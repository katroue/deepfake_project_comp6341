# Training Strategies

All strategies train an **EfficientNet-B1** classifier on the FaceForensics++ dataset (c23 compression) to perform binary real/fake detection. Each strategy targets a different hypothesis for improving generalisation.

---

## Shared Setup

| Parameter | Value |
|---|---|
| Model | EfficientNet-B1 (6.52M parameters) |
| Dataset | FaceForensics++ c23 — 193,980 train / 36,780 val samples |
| Manipulations | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| Optimizer | AdamW |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | patience=5 (except Strategy 3: disabled) |

---

## Strategy 1 — Baseline

**File:** `src/training/train_baseline.py`
**Config:** `configs/c23/strategy1_baseline.yaml`

### Description
Standard supervised fine-tuning of a pretrained EfficientNet-B1. Serves as the performance floor that all other strategies are compared against.

### Key Parameters
| Parameter | Value |
|---|---|
| Pretrained | ImageNet weights |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Augmentation | Baseline (resize, normalise, horizontal flip) |

### How it works
1. Load ImageNet-pretrained EfficientNet-B1
2. Replace classifier head with a 2-class output
3. Train all layers end-to-end for up to 20 epochs
4. Save best checkpoint by val accuracy

### Hypothesis
A well-initialised pretrained backbone with minimal interference should provide a strong baseline. Any strategy that fails to beat this is not adding value.

---

## Strategy 2 — Heavy Augmentation

**File:** `src/training/train_augmented.py`
**Config:** `configs/c23/stategy2_augmented.yaml`

### Description
Identical architecture and training loop to Strategy 1, but applies aggressive data augmentation to improve robustness to real-world variation in face images.

### Key Parameters
| Parameter | Value |
|---|---|
| Pretrained | ImageNet weights |
| Learning rate | 0.0001 |
| Augmentation | Heavy (see below) |

### Augmentation Pipeline
| Transform | Parameters |
|---|---|
| Random rotation | ±15° |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 |
| Random affine | translate=(0.1, 0.1), scale=(0.9, 1.1) |
| Random perspective | distortion_scale=0.2, p=0.5 |
| Random erasing | p=0.3, scale=(0.02, 0.1) |

### Hypothesis
Deepfake detectors are known to overfit to low-level compression artefacts. Aggressive augmentation forces the model to rely on higher-level semantic cues, improving generalisation to unseen manipulations.

---

## Strategy 3 — Curriculum Learning

**File:** `src/training/train_curriculum.py`
**Config:** `configs/c23/strategy3_curriculum.yaml`

### Description
Progressively expands the training distribution from easy, visually distinct manipulations to harder, more subtle ones. The validation set always uses all four manipulation types.

### Curriculum Schedule
| Phase | Epochs | Manipulation Types |
|---|---|---|
| Phase 1 | 1–6 | Face2Face only |
| Phase 2 | 7–11 | Face2Face, FaceSwap |
| Phase 3 | 12–16 | Face2Face, FaceSwap, Deepfakes |
| Phase 4 | 17–20 | Face2Face, FaceSwap, Deepfakes, NeuralTextures |

### Key Parameters
| Parameter | Value |
|---|---|
| Pretrained | ImageNet weights |
| Learning rate | 0.0001 |
| Augmentation | Baseline |
| Early stopping | Disabled (patience=20) — curriculum must complete all phases |

### How it works
At the start of each epoch, `CurriculumTrainer` rebuilds the training `DataLoader` to include only the manipulation types allowed in the current phase. The model first learns to distinguish the most visually obvious fake type (Face2Face), then incrementally adds harder examples.

### What to expect
Val accuracy will appear low or declining during phases 1–3 because the val set includes manipulation types the model has not yet seen. Meaningful val accuracy is only interpretable after phase 4 begins (epoch 17+).

### Hypothesis
Presenting easy examples first allows the model to form robust low-level features before being exposed to harder, more subtle manipulations — analogous to how humans learn.

---

## Strategy 4 — Self-Supervised Learning (SimCLR)

**File:** `src/training/train_ssl.py`
**Config:** `configs/c23/strategy4_ssl.yaml`

### Description
Two-phase training. Phase 1 uses contrastive self-supervised learning (SimCLR) to pretrain a backbone on all face images without labels. Phase 2 fine-tunes the pretrained backbone on the labelled FF++ dataset.

### Phase 1 — SimCLR Contrastive Pretraining

| Parameter | Value |
|---|---|
| Method | SimCLR (NT-Xent loss) |
| Backbone | EfficientNet-B1 (no classifier, trained from scratch) |
| Projection head | 2-layer MLP → 128-dim |
| Temperature | 0.5 |
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Augmentation | Strong (SimCLR-style: crop, flip, colour jitter, grayscale, blur) |

For each image, two randomly augmented views are generated. The model is trained to maximise agreement between the two views of the same image (positive pair) while pushing apart views from different images (negative pairs) using the NT-Xent loss.

### Phase 2 — Supervised Fine-Tuning

| Parameter | Value |
|---|---|
| Backbone init | Phase 1 pretrained weights (506 layers loaded) |
| Freeze backbone | First 3 epochs (classifier head only) |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Augmentation | Baseline |

The pretrained backbone is loaded and a 2-class classifier head is attached. For the first 3 epochs only the head is trained; after that all layers are unfrozen and trained end-to-end.

### Note on Phase 2 train vs val accuracy
During the frozen epochs (1–3), val accuracy can exceed train accuracy. This is because train accuracy is averaged over the full epoch (including early batches where the head is less trained), while validation runs on the fully-updated model after each epoch.

### Hypothesis
SSL pretraining forces the model to learn general, label-agnostic face representations. These representations may be more robust to manipulation artefacts than supervised ImageNet features, because the model learns to understand faces rather than object categories.

---

## Strategy 5 — Hard Negative Mining

**File:** `src/training/train_hard_neg.py`
**Config:** `configs/c23/strategy5_hard_neg.yaml`

### Description
Trains with standard uniform sampling for a warmup period, then periodically identifies low-confidence samples (hard negatives) and upsamples them in subsequent epochs.

### Key Parameters
| Parameter | Value |
|---|---|
| Pretrained | ImageNet weights |
| Learning rate | 0.0001 |
| Augmentation | Baseline |
| Warmup period | 5 epochs (uniform sampling) |
| Mining frequency | Every 5 epochs |
| Hard sample ratio | 50% of each batch |

### Adaptive Confidence Threshold
| Epoch | Threshold |
|---|---|
| 5 | 0.50 |
| 10 | 0.40 |
| 15 | 0.30 |
| 20 | 0.20 |

A sample is "hard" if the model's predicted class probability is below the threshold. The threshold tightens over time to keep focusing on genuinely difficult examples as the model improves.

### How it works
1. **Epochs 1–4:** Train with standard random sampling
2. **Epoch 5+:** Pass the full training set through the model, compute per-sample confidence scores, assign higher sampling weights to hard samples using `WeightedRandomSampler`
3. Re-mine every 5 epochs to update which samples are currently hard

### Hypothesis
Easy examples dominate uniform sampling and contribute little to learning. Focusing training compute on samples the model currently finds difficult accelerates learning of subtle manipulation cues.

---

## Strategy 6 — Multi-Task Learning

**File:** `src/training/train_multitask.py`
**Config:** `configs/c23/strategy6_multitask.yaml`

### Description
Jointly trains two classification heads on a shared EfficientNet-B1 backbone: a binary head (real/fake) and a 5-class manipulation-type head (Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures).

### Architecture
```
EfficientNet-B1 backbone
        │
   shared features
   ┌────┴────┐
Binary head  Multiclass head
(real/fake)  (5 manipulation types)
```

### Loss Function
```
total_loss = 0.7 × binary_loss + 0.3 × multiclass_loss
```

| Task | Weight | Classes |
|---|---|---|
| Binary (primary) | 0.7 | Real, Fake |
| Multiclass (auxiliary) | 0.3 | Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures |

### Key Parameters
| Parameter | Value |
|---|---|
| Pretrained | ImageNet weights |
| Learning rate | 0.0001 |
| Augmentation | Baseline |
| Checkpoint metric | Val binary accuracy |

### How it works
Each batch produces predictions from both heads. The combined weighted loss is backpropagated through the shared backbone, so gradients from the manipulation-type task regularise the shared features learned for binary detection.

### Hypothesis
Forcing the model to simultaneously identify *which* manipulation technique was used requires learning richer, more discriminative features than binary detection alone. The auxiliary task acts as a regulariser and should improve generalisation of the primary binary task.

---

## Strategy Comparison Summary

| Strategy | Key Idea | Pretrained | Augmentation |
|---|---|---|---|
| 1 — Baseline | Standard fine-tuning | ImageNet | Minimal |
| 2 — Heavy Augmentation | Robustness via transforms | ImageNet | Heavy |
| 3 — Curriculum Learning | Easy → hard manipulation schedule | ImageNet | Minimal |
| 4 — SSL (SimCLR) | Self-supervised face representations | None (scratch) | Strong (Phase 1) |
| 5 — Hard Negative Mining | Oversample difficult examples | ImageNet | Minimal |
| 6 — Multi-Task Learning | Auxiliary manipulation-type task | ImageNet | Minimal |
