# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Name:** Muhammad Ismail  
**Registration number:** 026  
**Dataset:** BloodMNIST, size 64  
**Total marks:** 25

This report presents the results and explanations from the executed Colab notebook.

## Prediction recorded before the experiments

I predicted HOG would lose more accuracy under rotation because rotating the cell changes gradient directions and their positions within fixed cells. I also predicted HOG would lose more accuracy under Gaussian noise because noise introduces additional local gradients, while SIFT's descriptor smoothing may reduce some of their effect. These were hypotheses; fixed-grid SIFT was not assumed to be rotation invariant.

## Q1. Data setup 

Registration 026 gives a random seed of **26**, HOG cells of **8 × 8**, and a SIFT grid of **6 × 6**. Images were selected from the official train, validation and test splits, converted to grayscale, resized to **128 × 128**, and scaled to **[0, 1]**.

| Split | Eosinophil (1) | Monocyte (5) | Neutrophil (6) | Platelet (7) | Total | Image array shape |
|---|---:|---:|---:|---:|---:|---|
| Training | 100 | 100 | 100 | 100 | 400 | (400, 128, 128) |
| Validation | 30 | 30 | 30 | 30 | 120 | (120, 128, 128) |
| Test | 30 | 30 | 30 | 30 | 120 | (120, 128, 128) |


### Four training examples per class

![Four training examples per class](labMid/md&visualizations/figure_01.png)

## Q2. HOG + SVM 

HOG summarizes local gradient directions, describing edges and intensity patterns. The extractor used **9 orientation bins**, **8 × 8 pixels per cell**, **2 × 2 cells per block**, and **L2-Hys block normalization**.

### Feature length: hand calculation

- Cells per side: 128 ÷ 8 = **16**.
- Overlapping block positions per side: 16 − 2 + 1 = **15**.
- Values per block: 2 × 2 × 9 = **36**.
- Total feature length: 15 × 15 × 36 = **8,100**.

The notebook printed a training feature shape of **(400, 8100)**, confirming the calculation.

### HOG visualization

The top row contains one training image per class; the bottom row shows its HOG visualization. Bright strokes indicate strong local gradients.

![Original images and HOG visualizations](0labMid/md&visualizations/figure_02.png)

### Validation and classifier selection

Feature scaling was fitted on training data only. LinearSVC was trained with each required C value and selected using validation accuracy. Ties retained the smaller C.

|  | C | validation accuracy | macro-F1 |
| --- | --- | --- | --- |
| 0 | 0.1 | 0.733333 | 0.73165 |
| 1 | 1.0 | 0.733333 | 0.73165 |
| 2 | 10.0 | 0.733333 | 0.73165 |

The selected C was **0.1**. Validation accuracy was **73.33%** (88/120), and macro-F1 was **0.7316**. All three C values gave the same validation predictions' aggregate scores.

![HOG validation confusion matrix](0labMid/md&visualizations/figure_03.png)

Rows show true labels and columns show predicted labels. Diagonal entries are correct classifications; off-diagonal entries are mistakes.

## Q3. SIFT + SVM

### Normal SIFT detection

Normal SIFT detection was run on all **400 training images**.

| Statistic | Observed result |
|---|---:|
| Mean keypoints per image | 72.28 |
| Minimum keypoints | 10 |
| Maximum keypoints | 188 |
| Images with zero keypoints | 0 |

![Normal SIFT keypoints on a training image](0labMid/md&visualizations/figure_04.png)

Different numbers of keypoints produce different-length concatenated descriptor vectors, but an SVM requires every image to have the same number of input features.

### Fixed-grid solution and feature length

A regular **6 × 6 grid** supplied **36 keypoints**. SIFT computed a **128-value descriptor** at each grid point, and descriptors were joined in a consistent order.

**Feature length = 6 × 6 × 128 = 4,608.**

The printed training feature shape was **(400, 4608)**.

### Validation and classifier selection

The same training-only scaling and validation selection procedure was applied to SIFT features.

|  | C | validation accuracy | macro-F1 |
| --- | --- | --- | --- |
| 0 | 0.1 | 0.708333 | 0.708303 |
| 1 | 1.0 | 0.716667 | 0.717466 |
| 2 | 10.0 | 0.716667 | 0.717466 |

The selected C was **1**. Validation accuracy was **71.67%** (86/120), and macro-F1 was **0.7175**.

![SIFT validation confusion matrix](0labMid/md&visualizations/figure_05.png)

## Q4. Robustness 

Both validation-selected models were frozen. Each was evaluated on the same **120 test images** under three conditions: clean, rotated by **20 degrees**, and Gaussian noise with **σ = 0.05**.

![Clean, rotated and noisy versions of the same test image](0labMid/md&visualizations/figure_06.png)

### Test results

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---:|---:|---:|
| HOG + SVM | 77.50% (93/120) | 67.50% (81/120) | 25.00% (30/120) |
| SIFT + SVM | 78.33% (94/120) | 73.33% (88/120) | 78.33% (94/120) |


## Conclusion

HOG had the higher validation accuracy, but SIFT had slightly higher clean-test accuracy and smaller losses under both perturbations. For the sampled test images and settings used here, **SIFT + SVM was the more robust pipeline**.
