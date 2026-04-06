# Strategy 4 — Self-Supervised Learning: Investigation Log

## Overview

Strategy 4 was designed as a two-phase self-supervised learning approach:
- **Phase 1:** Pretrain a backbone on all face images (real + fake) without labels
- **Phase 2:** Supervised fine-tuning on the labelled FF++ dataset using the pretrained backbone

Two methods were attempted: **SimCLR** (contrastive learning) and **SimSiam** (self-supervised without negative pairs). Both failed to produce a useful classifier, collapsing to predicting all images as fake.

**Final test result:** Accuracy 78.43% | AUC 0.5013 | Recall 1.0 | FNR 0.0

---

## Attempt 1 — SimCLR (NT-Xent Contrastive Loss)

### Method
SimCLR generates two randomly augmented views of each image and trains the backbone to maximise agreement between the two views of the same image (positive pair) while pushing apart views from different images (negative pairs) using the NT-Xent loss.

### Configuration
| Parameter | Value |
|---|---|
| Backbone | EfficientNet-B1 (from scratch) |
| Projection head | 2-layer MLP → 128-dim |
| Temperature | 0.5 |
| Batch size | 32 → 64 (adjusted for hardware) |
| Phase 1 epochs | 10 → 50 (increased after crash) |
| Hardware | Kaggle T4 GPU → local MPS (Apple Silicon) |

### Failure 1: Training crash (Kaggle session timeout)
The original run (10 epochs) was interrupted by a Kaggle session timeout before Phase 1 completed. Phase 2 started with a partially trained or corrupted backbone, causing the model to collapse to predicting all-fake (AUC ≈ 0.49).

**Fix attempted:** Increased Phase 1 epochs to 50 and added per-epoch checkpoint saving with resume support (`phase1_resume.pth`).

### Failure 2: NaN loss at epoch 10 (MPS float16 instability)
When training locally on Apple Silicon (MPS), the NT-Xent loss went NaN starting at epoch 10. The cause was MPS float16 instability in the contrastive loss computation — the `-inf` diagonal masking combined with reduced-precision arithmetic caused NaN to propagate through the similarity matrix.

**Fix applied:** Disabled `torch.autocast` for MPS (kept for CUDA only), added gradient clipping (`max_norm=1.0`).

### Failure 3: Loss plateau — SimCLR fundamentally requires large batches
After the NaN fix, Phase 1 trained for 20 epochs but the loss flatlined after epoch 3:

| Epoch | Loss |
|---|---|
| 1 | 3.1081 |
| 2 | 2.9254 |
| 3 | 2.9029 |
| 4–20 | 2.87–2.89 (flat) |

**Root cause:** SimCLR's NT-Xent loss quality scales directly with batch size — more negatives per step means a stronger learning signal. With MPS memory constraints limiting batch size to 64 (126 negatives per sample), the model quickly saturated what it could learn and plateaued. The SimCLR paper uses batch sizes of 256–8192. Phase 2 fine-tuning on a backbone that learned nothing useful predictably collapsed to all-fake.

---

## Attempt 2 — SimSiam

### Why SimSiam
SimSiam does not use negative pairs. Each training step only compares two augmented views of the same image, with a stop-gradient on one branch to prevent representation collapse. This makes it **batch-size independent** — the quality of the learning signal does not depend on how many other samples are in the batch.

### Method
Two augmented views are passed through a shared backbone + projector. A predictor head on one branch predicts the other branch's projection (stop-gradient applied). Loss is negative cosine similarity, ranging from −1 (perfect) to 0 (random).

### Configuration
| Parameter | Value |
|---|---|
| Backbone | EfficientNet-B1 (from scratch) |
| Projector | 3-layer MLP with BN → 512-dim |
| Predictor | 2-layer MLP → 128-dim hidden → 512-dim |
| Batch size | 64 (96 caused memory pressure on MPS) |
| Phase 1 epochs | 20 |
| Hardware | Kaggle T4 GPU |
| Phase 1 loss (epoch 20) | −0.9888 (converged well, approaching −1) |

### Phase 1 — Successful convergence
Unlike SimCLR, SimSiam converged meaningfully on Kaggle's T4 GPU:

| Epoch | Loss |
|---|---|
| 1 | −0.9208 |
| 5 | −0.9718 |
| 10 | −0.9780 |
| 15 | −0.9805 |
| 20 | −0.9888 |

Loss steadily approached −1 (perfect alignment), confirming the backbone learned useful self-supervised representations of face images.

### Phase 2 — Collapse to all-fake

Despite a well-converged Phase 1 backbone, Phase 2 fine-tuning failed. Val accuracy was locked at exactly 78.24% (the all-fake baseline) for all 6 epochs before early stopping triggered.

Several issues were diagnosed and fixed during Phase 2:

**Issue 1: Pretrained backbone not loading**
The `pretrained_path` in the config was a relative path (`results/models/...`) while the file was copied to an absolute Kaggle path. The `os.path.exists()` check failed silently, and Phase 2 trained from random weights. Fixed by explicitly setting the absolute path in the config and fixing `main()` to fall back to `config['phase_2']['pretrained_path']` when Phase 1 is disabled.

**Issue 2: BatchNorm running statistics mismatch**
Phase 1 used strong SSL augmentations (colour jitter, grayscale, Gaussian blur). The BatchNorm running mean/variance accumulated during Phase 1 were calibrated for this heavily augmented distribution. In Phase 2 eval mode, PyTorch uses these stale running statistics instead of batch statistics — causing the model to apply incorrect normalisation to unaugmented validation images and produce degenerate outputs (all-fake). Fixed by resetting all BN running stats after loading the pretrained weights:
```python
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.reset_running_stats()
```

**Issue 3: Class weight instability**
With `class_weight_real: 3.64` (matching the exact dataset imbalance ratio), the weighted loss caused oscillation — the model alternated between predicting all-real and all-fake, settling at ~50% training accuracy. Reduced to `class_weight_real: 2.0` to stabilise training. Training accuracy then climbed steadily (59% → 64%) but val accuracy remained at 78.24%.

### Root cause of Phase 2 failure

Despite fixing the above issues, the model consistently predicted all images as fake during validation. The train/eval accuracy gap (64% train vs 78% val) persisted even after the BN reset.

**The fundamental problem:** SimSiam trains a backbone to learn *augmentation-invariant* representations — features that are stable across random crops, colour shifts, and blurs. Deepfake detection, however, relies on precisely the opposite: subtle, low-level compression artefacts and blending inconsistencies that are *specific to the manipulation process* and easily suppressed by augmentation.

By design, SimSiam's pretraining encourages the backbone to ignore exactly the kinds of signals that make deepfakes detectable. The resulting backbone, despite converging well on the SSL objective, produced representations that were not useful for real vs. fake discrimination. Phase 2 fine-tuning could not recover discriminative features from this starting point.

---

## Summary

| Attempt | Phase 1 outcome | Phase 2 outcome | Root cause of failure |
|---|---|---|---|
| SimCLR | Loss plateau (2.87, flat from epoch 3) | All-fake collapse | Batch size too small for NT-Xent loss (MPS constraint: max 64) |
| SimSiam | Converged well (loss −0.99) | All-fake collapse | SSL objective learns augmentation-invariant features, suppressing deepfake artefacts |

## Conclusion

SSL pretraining is fundamentally misaligned with deepfake detection. Deepfake artefacts (compression boundaries, blending seams, GAN fingerprints) are subtle, low-frequency signals that augmentation-based self-supervised methods are trained to discard as noise. A backbone pretrained to be invariant to these signals cannot be fine-tuned to detect them — the information has been removed during pretraining.

All five supervised strategies (Baseline through Multi-Task) achieved 84–88% test accuracy by training directly on labelled real/fake examples, confirming that label supervision is necessary for this task. Future work could explore supervised contrastive learning (SupCon), which uses labels during the contrastive phase and would avoid this misalignment.
