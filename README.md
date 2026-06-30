# Development_tools_for_AI_classification_in_reproductive_biology

**Scripts and models for Python.**

**Dr. Tomáš Venit** from the [Laboratory of Biology of the Cell Nucleus](https://www.img.cas.cz/group/pavel-hozak/), [Institute of Molecular Genetics of the Czech Academy of Sciences](https://www.img.cas.cz/en/) in Prague, Prague, Czech Republic is the PI of the MEYS–LM2023050 Czech-BioImaging project “Development of tools for AI classification in reproductive biology” and contributed to this study.

Also acknowledged by the Light Microscopy Core Facility, IMG, Prague, Czech Republic, supported by MEYS – LM2023050 Czech-BioImaging, MEYS – CZ.02.1.01/0.0/0.0/18_046/0016045, and MEYS – CZ.02.01.01/00/23_015/0008205 and by institutional support from the Institute of Molecular Genetics of the Czech Academy of Sciences (RVO: 68378050) and the
Research Program Strategy AV21 Future of Assisted Reproduction (ART) (AV21-VP38/2025) provided by the Czech Academy of Sciences. Anonymized images of human embryos were provided by Clayo Clinic, Prague.

## Overview

Accurate segmentation and quantification of cells in light microscopy images remain essential yet challenging tasks in reproductive biology. Manual annotation of embryonic cells is labor-intensive, subjective, and difficult to scale for larger datasets. In this work, we present a deep learning-based pipeline for automated detection, segmentation, and quantification of human embryonic cells from microscopy images acquired during early developmental stages.  

## Methods

### Models

The proposed approach combines **a count prediction network** with **a slot-based instance segmentation model**. A retrained **ResNet18 classifier** first **predicts the number of cells** present in an embryo image, while a second model based on a **ResNet18 encoder and U-Net decoder** performs **instance segmentation** using four fixed output slots with permutation-invariant training, see Fig. 1 and a **poster** attached.

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

### Permutation-Invariant Training

Because instance ordering is arbitrary, slot assignments are treated as unordered.

During training:
1. All possible assignments between predicted slots and ground-truth slots are evaluated.
2. Binary Cross-Entropy (BCE) and Dice losses are computed for each assignment.
3. The assignment with the lowest total loss is selected.

This ensures that the model learns instance segmentation independently of slot order.

### Count-Guided Inference

#### Count Classifier
The count classifier predicts the number of valid instances: **K ∈ {1, 2, 4}**.

#### Postprocessing
Each predicted slot undergoes:
- Sigmoid activation
- Thresholding
- Largest connected component extraction
- Area and confidence filtering

#### Top-K Selection
Based on the predicted count \(K\), only the best \(K\) instance masks are retained, while remaining slots are discarded.

### Final Outputs

The pipeline produces:

1. **Instance label image** (labels 0–4)
2. **Colorized overlay** of retained instances
3. **Ellipse fits** derived from final segmentation masks
4. **JSON export** containing ellipse parameters

### Key Characteristics

- Shared encoder-decoder architecture
- Four fixed segmentation slots
- Permutation-invariant loss removes dependence on slot ordering
- Count-guided selection suppresses false-positive instances
- Supports variable numbers of embryos without requiring explicit slot identities

## Manual Annotations

To simplify annotation and provide biologically interpretable outputs, embryonic cells were represented by **ellipses manually annotated in Fiji**, Fig. 2. During training, ellipse annotations were converted into polygon masks to obtain pixel-level supervision, Fig. 3.

**Figure 2:** Examples (1-, 2-, 4-cells) of manual annotations using Fiji and Elliptical selection tool.
<img width="2148" height="660" alt="Figure_2" src="https://github.com/user-attachments/assets/ed9227df-f723-4b57-b743-c7873113a011" />

**Figure 3:** Conversion of ellipse annotations into pixel-wise polygon masks.
<img width="1657" height="549" alt="Figure_3" src="https://github.com/user-attachments/assets/ebb84317-90cf-4068-9a2b-f7f154c6e2a4" />

The training dataset consisted of **180 annotated embryo images equally distributed among one-cell, two-cell, and four-cell** developmental stages. Data augmentation included rotation, scaling, and shifting transformations, while optimization employed a combined Dice and binary cross-entropy loss. During inference, segmentation candidates are filtered according to the predicted cell count using a top-K selection strategy, followed by ellipse fitting to generate the final representation.  

## Inference

The workflow combines a deep learning–based cell count classifier with a slot-based instance segmentation network, Fig. 4. A ResNet18 classifier first predicts the number of cells present in the input image (K∈{1,2,4}). In parallel, a ResNet18 encoder–U-Net decoder segmentation network produces four fixed output slots, each representing a candidate cell mask. The K most relevant masks are selected according to the classifier prediction, and an ellipse is fitted to each selected mask using contour-based ellipse fitting. The resulting ellipse representations provide compact, interpretable descriptions of individual cells while suppressing spurious detections and enabling robust quantitative analysis.

**Figure 4:** Overview of the proposed embryo cell analysis pipeline.
<img width="9504" height="2406" alt="Figure_4-Smaller" src="https://github.com/user-attachments/assets/3c0446df-3b83-4c53-8487-9e881808f920" />

## Results

Performance of the pipeline was evaluated on **1,022 microscopy images** and compared in three scenarios, Fig. 5. **Left:** the ResNet18-based cell count classifier alone achieves 99.9% accuracy, demonstrating highly reliable prediction of the number of embryonic cells. **Center:** the slot-based segmentation network alone reaches 96.97% counting accuracy, with most errors caused by over-segmentation of two-cell embryos. **Right:** combining the count classifier with the segmentation network by selecting the top-K predicted masks according to the estimated cell count increases the overall detection accuracy to 99.61%. In addition, the fitted ellipse representation closely matches the predicted segmentation masks, achieving a **mean Dice coefficient of 0.981** and a **mean IoU of 0.964**, confirming that ellipse fitting provides an accurate and compact geometric representation of segmented embryonic cells, Fig. 6.

**Figure 5:** Quantitative evaluation of the proposed pipeline.
<img width="3682" height="856" alt="Figure_5" src="https://github.com/user-attachments/assets/3d7d6f2f-daed-44df-a49d-55c2e73b036c" />

**Figure 6:** Examples of segmentation of 1-, 2-, 4-cells. From left: input image, predicted probability maps, thresholded binary masks, mask overlay, and fitted ellipses. The count-aware model first predicts the cell number, then selects the top-K segmentation candidates and fits ellipses to each detected cell.
<img width="2565" height="1323" alt="Figure_6" src="https://github.com/user-attachments/assets/8a6a7996-de0d-4035-aaad-2ceb07b4ff9f" />

## Codes

### 01_cellcount_cnn_vscode_verbose_v2.py

The **first** script trains and evaluates a convolutional neural network (ResNet18 or EfficientNet-B0) to classify human embryo microscopy images into 1-, 2-, or 4-cell stages, and supports subsequent inference on new images.

**Input.** The pipeline processes 8-bit grayscale microscopy images (400 × 400 pixels) of human embryos. The number of cells is encoded in the image filename, allowing automatic assignment of class labels (1-, 2-, or 4-cell embryos). The dataset is split at the embryo level to prevent data leakage between training, validation, and testing. 

**Processing.** Images are converted to three-channel inputs and normalized before being passed through a CNN classifier (ResNet18 or EfficientNet-B0). During training, extensive data augmentation is applied, including random flips, rotations, affine transformations, intensity variations, and Gaussian blur. The pipeline supports pretrained models, class balancing via weighted loss or oversampling, mixed-precision training, early stopping, learning-rate scheduling, and comprehensive logging. Model performance is evaluated using accuracy, macro F1-score, classification reports, and confusion matrices. 

**Output.** The trained model predicts the embryo cell count (1, 2, or 4 cells) together with class probabilities. During training, the pipeline saves the best-performing model checkpoint, validation reports, confusion matrices, and final evaluation statistics. In inference mode, predictions for all processed images are exported to a CSV file containing the predicted class and associated confidence scores. 

### 02_export_fiji_rois_to_ellipses.py

The **second** script converts Fiji/ImageJ ROI annotations of embryo cells into ellipse-based JSONL annotations and generates optional quality-control overlay images for training and validation.

**Input.** The script takes grayscale TIFF microscopy images together with corresponding Fiji/ImageJ ROI annotations, where each ROI represents a manually annotated embryo cell. The expected number of cells is automatically extracted from the image filename and verified against the number of ROI files. 

**Processing.** Each ROI is converted into a parametric ellipse by fitting its contour or, for oval ROIs, by using its bounding box. The extracted ellipse parameters (center coordinates, semi-major axis, semi-minor axis, and orientation) are stored in a JSONL annotation file. Optionally, quality-control overlay images with fitted ellipses and labels are generated for visual verification. 

**Output.** The script produces a JSONL file containing ellipse-based annotations for all images and, optionally, overlay images visualizing the fitted ellipses on the original microscopy images to facilitate annotation quality assessment. 

### 03_M6_train_segmentation.py

The **third** script trains and validates a ResNet18-based U-Net for count-aware slot-based embryo cell segmentation using ellipse annotations, geometry-consistent data augmentation, and permutation-invariant loss, while automatically saving the best-performing models and evaluation results.

**Input.** The script uses grayscale microscopy images together with ellipse-based JSONL annotations generated from Fiji/ImageJ ROIs. The dataset is automatically split into training and validation subsets while preserving the distribution of embryo cell counts. 

**Processing.** A ResNet18-based U-Net is trained to predict four fixed segmentation slots corresponding to potential embryo cells. Training incorporates geometry-consistent data augmentation (rotation, scaling, translation, and flipping), permutation-invariant BCE + Dice loss, early stopping, and learning-rate scheduling. Model performance is monitored using validation Dice scores, loss curves, per-class metrics, and qualitative prediction previews. 

**Output.** The script saves the best-performing segmentation models, training statistics, learning curves, validation preview images, and a JSON file containing the complete training history and evaluation metrics. 

### 04_M9_inference_for_unknown_data_full_export_v2.py

The **fourth** script performs fully automated inference on previously unseen embryo images by combining cell-count classification with slot-based segmentation, exporting ellipse-based cell annotations together with comprehensive visualization and evaluation reports.

**Input.** The script processes previously unseen grayscale microscopy images using pretrained cell-count classification and slot-based segmentation models. If available, the expected cell count is extracted from the image filename for evaluation purposes. 

**Processing.** A count classifier first predicts the number of embryo cells (1, 2, or 4), after which a ResNet18-based U-Net generates four candidate segmentation masks. The top-*K* masks are selected according to the predicted count, converted into fitted ellipses, and used to create instance label images, visualization overlays, and quantitative performance summaries. 

**Output.** The script exports ellipse annotations (JSON), overlay and instance-label images, summary CSV/JSON reports, confusion matrices, mismatch logs, and evaluation statistics, including classification performance and mask-to-ellipse self-consistency metrics. 

## Trained models and example images

vysvětlit obrazky, čísla, časy, foc=preprocessing, stručně popsat kroky preprocessingu, další čísla 1,2,4, orig vs. další obrázky

## How to use the codes and the models

vs code, upravit cesty k obrázkům a modelům, včetně názvů modelů, připravit python prostředí - yml soubor
