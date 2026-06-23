# Development_tools_for_AI_classification_in_reproductive_biology

**Scripts and models for Python.**

**Dr. Tomáš Venit** from the [Laboratory of Biology of the Cell Nucleus](https://www.img.cas.cz/group/pavel-hozak/), [Institute of Molecular Genetics of the Czech Academy of Sciences](https://www.img.cas.cz/en/) in Prague, Prague, Czech Republic is the PI of the MEYS–LM2023050 Czech-BioImaging project “Development of tools for AI classification in reproductive biology” and contributed to this study.

Also acknowledged by the Light Microscopy Core Facility, IMG, Prague, Czech Republic, supported by MEYS – LM2023050 Czech-BioImaging, MEYS – CZ.02.1.01/0.0/0.0/18_046/0016045, and MEYS – CZ.02.01.01/00/23_015/0008205 and by institutional support from the Institute of Molecular Genetics of the Czech Academy of Sciences (RVO: 68378050) and the
Research Program Strategy AV21 Future of Assisted Reproduction (ART) (AV21-VP38/2025) provided by the Czech Academy of Sciences. Anonymized images of human embryos were provided by Clayo Clinic, Prague.

## Overview

Accurate segmentation and quantification of cells in light microscopy images remain essential yet challenging tasks in reproductive biology. Manual annotation of embryonic cells is labor-intensive, subjective, and difficult to scale for larger datasets. In this work, we present a deep learning-based pipeline for automated detection, segmentation, and quantification of human embryonic cells from microscopy images acquired during early developmental stages.  

## Methods

### Models

The proposed approach combines **a count prediction network** with **a slot-based instance segmentation model**. A retrained **ResNet18 classifier** first **predicts the number of cells** present in an embryo image, while a second model based on a **ResNet18 encoder and U-Net decoder** performs **instance segmentation** using four fixed output slots with permutation-invariant training, see Fig. 1.

**Figure 1:** Schematic diagram of a slot-based instance segmentation framework designed for detecting and segmenting up to four embryo instances in grayscale microscopy images.
<img width="11250" height="6125" alt="architecture_slots_diagram_journal_grade_MC" src="https://github.com/user-attachments/assets/22f4f2a0-e785-43f0-bfc3-34b0ad7542b7" />

## Architecture

### Input and Supervision
- **Input:** Grayscale embryo image (`1 × 400 × 400`).
- **Ground Truth:** Up to four instance masks generated from annotated ellipses and assigned to four unordered ground-truth slots.

### Shared Segmentation Network
The segmentation backbone consists of:
- **Encoder:** ResNet18-style feature extractor.
  - Conv1 (64 channels)
  - Layer1 (64 channels)
  - Layer2 (128 channels)
  - Layer3 (256 channels)
  - Layer4 (512 channels)
- **Bottleneck:** Center block (512 channels)
- **Decoder:** U-Net decoder with skip connections.
  - UpBlock1 (256 channels)
  - UpBlock2 (128 channels)
  - UpBlock3 (64 channels)
  - UpBlock4 (64 channels)

A final **1×1 convolution** produces four output channels corresponding to four segmentation slots.

### Output Slots
The network predicts: **Slot 1** / **Slot 2** / **Slot 3** / **Slot 4**.  
Each slot outputs logits that are converted into an instance probability map.

## Permutation-Invariant Training

Because instance ordering is arbitrary, slot assignments are treated as unordered.

During training:
1. All possible assignments between predicted slots and ground-truth slots are evaluated.
2. Binary Cross-Entropy (BCE) and Dice losses are computed for each assignment.
3. The assignment with the lowest total loss is selected.

This ensures that the model learns instance segmentation independently of slot order.

## Count-Guided Inference

### Count Classifier
The count classifier predicts the number of valid instances: **K ∈ {1, 2, 4}**.

### Postprocessing
Each predicted slot undergoes:
- Sigmoid activation
- Thresholding
- Largest connected component extraction
- Area and confidence filtering

### Top-K Selection
Based on the predicted count \(K\), only the best \(K\) instance masks are retained, while remaining slots are discarded.

## Final Outputs

The pipeline produces:

1. **Instance label image** (labels 0–4)
2. **Colorized overlay** of retained instances
3. **Ellipse fits** derived from final segmentation masks
4. **JSON export** containing ellipse parameters

## Key Characteristics

- Shared encoder-decoder architecture
- Four fixed segmentation slots
- Permutation-invariant loss removes dependence on slot ordering
- Count-guided selection suppresses false-positive instances
- Supports variable numbers of embryos without requiring explicit slot identities

### Annotations

To simplify annotation and provide biologically interpretable outputs, embryonic cells were represented by ellipses manually annotated in Fiji. During training, ellipse annotations were converted into polygon masks to obtain pixel-level supervision.  

The training dataset consisted of 180 annotated embryo images equally distributed among one-cell, two-cell, and four-cell developmental stages. Data augmentation included rotation, scaling, and shifting transformations, while optimization employed a combined Dice and binary cross-entropy loss. During inference, segmentation candidates are filtered according to the predicted cell count using a top-K selection strategy, followed by ellipse fitting to generate the final representation.  

### Results

The standalone counting model achieved 99.9% accuracy on a test set of 1022 images, whereas the segmentation model alone reached 96.97% accuracy. Combining both approaches significantly improved effective detection performance, yielding 99.61% accuracy. These results demonstrate that integrating count-aware prediction with slot-based segmentation provides a robust and efficient framework for automated quantitative analysis of embryonic microscopy data.  
