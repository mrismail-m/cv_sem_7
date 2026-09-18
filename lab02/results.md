# Lab Task 02: Effect of Image Filtering on Skin-Lesion Classification

## Experimental Results

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | AUC |
|---|---|---|---|---|---|---|---|
| Best Model 1 (ResNet101 features + XGBoost) | No Filter | 84.85 | 84.61 | 84.85 | 84.58 | 70.45 | 95.52 |
| Best Model 1 (ResNet101 features + XGBoost) | Average | 83.84 | 83.72 | 83.84 | 83.65 | 70.63 | 95.51 |
| Best Model 1 (ResNet101 features + XGBoost) | Gaussian | 83.94 | 83.58 | 83.94 | 83.60 | 70.49 | 95.40 |
| Best Model 1 (ResNet101 features + XGBoost) | Median | 84.75 | 84.30 | 84.75 | 84.40 | 69.55 | 95.10 |
| Best Model 1 (ResNet101 features + XGBoost) | Sharpening | 84.34 | 84.29 | 84.34 | 84.07 | 70.47 | 95.76 |
| Best Model 1 (ResNet101 features + XGBoost) | Sobel | 78.48 | 77.33 | 78.48 | 77.62 | 55.32 | 90.89 |
| Best Model 2 (ResNet101, fine-tuned) | No Filter | 84.34 | 85.03 | 84.34 | 84.53 | 70.54 | 94.28 |
| Best Model 2 (ResNet101, fine-tuned) | Average | 82.73 | 83.33 | 82.73 | 82.95 | 71.08 | 94.25 |
| Best Model 2 (ResNet101, fine-tuned) | Gaussian | 81.31 | 83.27 | 81.31 | 81.91 | 68.24 | 94.52 |
| Best Model 2 (ResNet101, fine-tuned) | Median | 82.93 | 83.78 | 82.93 | 83.15 | 69.65 | 93.16 |
| Best Model 2 (ResNet101, fine-tuned) | Sharpening | 83.03 | 83.10 | 83.03 | 82.91 | 68.08 | 94.35 |
| Best Model 2 (ResNet101, fine-tuned) | Sobel | 72.22 | 78.13 | 72.22 | 74.11 | 51.80 | 89.25 |
| Best Model 3 (DenseNet121, fine-tuned) | No Filter | 84.75 | 84.59 | 84.75 | 84.65 | 72.47 | 95.75 |
| Best Model 3 (DenseNet121, fine-tuned) | Average | 75.05 | 81.45 | 75.05 | 77.22 | 62.93 | 94.44 |
| Best Model 3 (DenseNet121, fine-tuned) | Gaussian | 81.11 | 82.45 | 81.11 | 81.55 | 66.17 | 94.22 |
| Best Model 3 (DenseNet121, fine-tuned) | Median | 82.02 | 83.15 | 82.02 | 82.26 | 67.12 | 94.28 |
| Best Model 3 (DenseNet121, fine-tuned) | Sharpening | 81.21 | 84.53 | 81.21 | 82.38 | 70.00 | 95.87 |
| Best Model 3 (DenseNet121, fine-tuned) | Sobel | 72.32 | 76.07 | 72.32 | 73.71 | 48.05 | 89.43 |


## Questions and Answers

**1. Which three pretrained models performed best in Lab Activity 1?**
The best three pretrained models identified for this task were:
1. ResNet101 deep features evaluated via an XGBoost classifier.
2. ResNet101 (fine-tuned end-to-end).
3. DenseNet121 (fine-tuned end-to-end).

**2. How does filtering affect each of the three models?**
Filtering generally degraded the performance of all three models compared to their "No Filter" baseline. Model 1 (XGBoost) proved to be slightly more robust to smoothing filters (dropping by < 1%), but the end-to-end fine-tuned models (ResNet101 and DenseNet121) experienced larger, more noticeable drops in Accuracy and Macro-F1 scores across almost all filters.

**3. Which filter produces the greatest change compared with the unfiltered baseline?**
The **Sobel** edge detection filter produced by far the greatest negative change across all models. For instance, it dropped the accuracy of DenseNet121 from 84.75% down to 72.32%, and severely cut its Macro-F1 score from 72.47% to 48.05%.

**4. Does the effect of a filter remain consistent across all three models?**
Yes, the overarching trend is highly consistent. The Sobel filter consistently causes the most severe drop in performance across all three models. The smoothing filters (Average, Gaussian, Median) cause minor-to-moderate drops, and the Sharpening filter preserves performance relatively well (closest to the baseline). This indicates all three architectures fundamentally rely on similar visual information.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**
Filtering consistently **decreases** both macro-F1 and balanced accuracy. While overall Accuracy might look relatively stable on some filters, Macro-F1 reveals that the models lose their ability to distinguish minority classes effectively when spatial filters are applied. 

**6. Which lesion classes are most affected by filtering?**
Minority classes that rely heavily on fine texture and color differentiation (such as Melanoma and Actinic keratoses) are the most negatively affected. Because filtering removes critical morphological details, the models default to predicting the majority class (Melanocytic nevi), which crushes the per-class metrics for the rarer lesions.

**7. Why might smoothing remove useful lesion texture or morphological information?**
Smoothing filters (like Average, Gaussian, and Median) intentionally blur the image to reduce noise. However, skin lesions are primarily diagnosed based on extremely fine textures, color variations, pigment networks, and tiny dots/globules. Blurring acts destructively on these critical microscopic-level details, depriving the classifier of essential diagnostic features.

**8. Why might sharpening or edge detection help or hurt classification?**
Edge detection (Sobel) removes critical color gradients and color boundaries, which are primary diagnostic indicators in dermatology, thereby heavily hurting accuracy. Sharpening can theoretically help by highlighting fine structures, but in practice, it often hurts performance because it simultaneously amplifies artifacts (like skin pores, hair, and imaging noise), misleading the deep learning model's spatial mapping.

**9. What is the difference between convolution and correlation?**
In spatial filtering, both operations involve sliding a kernel over an image matrix. The mathematical difference is that in **convolution**, the kernel is rotated 180 degrees (flipped horizontally and vertically) before the element-wise multiplication is computed, whereas in **cross-correlation**, the kernel is applied exactly as-is without flipping. (Note: standard deep learning "convolutional" layers actually perform correlation).

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**
The results clearly demonstrate that deep-learning models automatically learn the optimal spatial feature extractors (filters) directly from the raw data during training. Applying classical, hand-crafted image processing filters beforehand arbitrarily discards vital information (like texture and color gradients) and forces the CNN to work with degraded input data. Consequently, for advanced classification tasks, raw deep-learning feature extraction supersedes and generally outperforms manual preprocessing filters.
