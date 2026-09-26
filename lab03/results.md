# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

## 1. Introduction
This report details the implementation and analysis of various edge detection techniques (Sobel, Prewitt, Laplacian, LoG, Canny) on the HAM10000 skin lesion dataset. It explores the sensitivity of these detectors to Gaussian and Salt-and-Pepper noise, evaluates the role of spatial smoothing, and compares the classification performance of traditional ML models and deep CNNs on raw, filtered, and edge-mapped images.

## 2. Experimental Setup
*   **Dataset**: HAM10000 (Skin Lesion Dataset)
*   **Edge Detectors**: Sobel (First-order), Prewitt (First-order), Laplacian (Second-order), Laplacian of Gaussian (LoG), Canny (Multi-stage).
*   **Noise Types**: Gaussian ($\sigma^2=0.01$), Salt-and-Pepper (5%).
*   **Classifiers**: SVM, Random Forest, KNN, ResNet101 (CNN 1), DenseNet121 (CNN 2).

## 3. Results and Observations

### Table 1: Effect of Noise and Preprocessing on Edge Detection
| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | Crisp, thick | Moderate | Captures main gradients well. |
| Sobel | Noisy | Gaussian | None | Degraded, noisy | High | Edges are lost in the background gradient noise. |
| Sobel | Noisy | Salt & Pepper | None | Severely degraded | High | S&P dots create massive false gradients. |
| Sobel | Noisy | Gaussian | Gaussian Filter | Restored, thicker | Low | Gaussian smoothing effectively counteracts Gaussian noise. |
| Sobel | Noisy | Salt & Pepper | Median Filter | Restored, clean | Low | Median filter perfectly removes S&P spikes before Sobel. |
| Prewitt | Original | None | None | Slightly softer than Sobel | Moderate | Very similar to Sobel but less emphasis on center pixels. |
| Laplacian | Original | None | None | Thin, noisy | Very High | Second-derivative amplifies minor texture variations. |
| LoG | Noisy | Gaussian | Gaussian Filter | Clean, closed loops | Low | The built-in Gaussian smoothing stabilizes the Laplacian. |
| Canny | Original | None | Built-in smoothing | Thin, connected | Low | Non-maximum suppression creates ideal 1-pixel wide edges. |
| Canny | Noisy | Gaussian | Gaussian Filter | Good, slightly shifted | Low | Pre-smoothing restores Canny's structural edge mapping. |
| Canny | Noisy | Salt & Pepper | Median Filter | Very clean | Low | Median filtering allows Canny to find true structural boundaries. |

### Table 2: Canny Parameter Analysis
| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observation |
|---|---|---|---|---|---|---|
| Canny-1 | 30 | 100 | 3×3 | Dense, cluttered | Very High | Captures too much internal skin texture and hair. |
| Canny-2 | 50 | 150 | 3×3 | Balanced, structural | Moderate | Good balance; isolates the main lesion boundary. |
| Canny-3 | 100 | 200 | 3×3 | Sparse, broken | Low | Misses softer lesion edges; boundaries are disconnected. |
| Canny-4 | 50 | 150 | 5×5 | Smooth, generalized | Low-Moderate | Larger kernel removes finer details, leaving only macro structures. |

*Selected Configuration*: **Canny-2 (50/150, 3x3)** produced the most useful representation by balancing noise rejection and boundary continuity.

### Table 3: Cross-Lab Classification Performance Comparison
| Model / Classifier | Accuracy Raw (Lab 1) | Accuracy Filtered (Lab 2) | Accuracy Edge (Lab 3) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| SVM | 73.2% | 68.5% | 45.1% | 46.2% | 45.1% | 44.8% |
| Random Forest | 76.5% | 71.2% | 48.3% | 49.1% | 48.3% | 47.9% |
| KNN | 68.4% | 63.8% | 38.2% | 39.5% | 38.2% | 38.6% |
| CNN Model 1 (ResNet101) | 84.3% | 82.7% | 55.4% | 56.1% | 55.4% | 55.2% |
| CNN Model 2 (DenseNet121) | 84.7% | 81.1% | 54.8% | 55.3% | 54.8% | 54.5% |

*(Note: Edge maps heavily degrade classification performance because crucial color and texture features are discarded.)*

## 4. Discussion Questions

**Question 1: Edge Detection and Noise**
*Which edge detector was most sensitive to noise? Explain your answer using your experimental observations.*
The **Laplacian** filter was the most sensitive to noise. Because it is a second-order derivative filter, it inherently amplifies high-frequency noise. In our experiments, applying the Laplacian to noisy images resulted in an unusable edge map dominated by isolated noise spikes rather than true structural edges.

**Question 2: Effect of Filtering**
*How did Gaussian and Median filtering affect the quality of detected edges?*
Gaussian filtering successfully smoothed out continuous Gaussian noise, allowing edge detectors to find continuous boundaries. Median filtering was exceptionally effective against Salt-and-Pepper noise, removing the outlier spikes entirely while preserving the sharpness of the actual lesion boundaries, leading to vastly superior edge maps.

**Question 3: Canny Parameters**
*How did changing the low and high thresholds affect the number and quality of detected edges?*
Lowering the thresholds (e.g., 30/100) resulted in a high number of detected edges, capturing excessive background texture (like skin pores and hair). Raising the thresholds (e.g., 100/200) strictly limited detection to strong gradients, reducing the number of edges but causing the main lesion boundaries to become broken and discontinuous.

**Question 4: Edge Maps and Classification**
*Did using edge-only images improve or reduce classification accuracy compared with raw images? Explain the possible reasons.*
Using edge-only images severely **reduced** classification accuracy. Skin lesions are primarily diagnosed using color variations, pigmentation networks, and internal texture. Edge detection aggressively discards all color and internal shading, leaving the classifier with only shape boundaries, which are often not unique enough to differentiate complex lesion classes.

**Question 5: Information Loss**
*Edge maps mainly represent object boundaries. What information may be lost when texture, color, and intensity information are removed?*
Color gradients (crucial for melanoma detection), global intensity distributions, and soft textures (like dots/globules) are completely lost. This information is vital for dermatological diagnosis, where the *color* and *fill* of the boundary are more diagnostic than the boundary shape itself.

**Question 6: Classical vs. Deep Features**
*CNNs can learn edge-like features automatically in their early layers. What are the advantages of allowing a CNN to learn these features instead of manually providing edge maps?*
Allowing a CNN to learn features automatically means the network will only extract edges that are *mathematically useful* for minimizing classification loss. It can learn color-specific edges, diagonal textures, and complex gradient combinations, whereas manual edge maps force a fixed, human-defined representation that discards potentially useful raw data.

**Question 7: Best Representation**
*Based on your results from Labs 01–03, which input representation produced the most useful classification results?*
**Raw images** produced the most useful classification results (yielding the highest accuracy and Macro-F1 scores). Filtering and edge detection discarded critical diagnostic information that deep learning models (like ResNet101 and DenseNet121) rely upon to make highly accurate predictions.

## 5. Conclusion
This laboratory demonstrated the mechanics of first, second, and multi-stage edge detectors. While tools like the Canny edge detector excel at isolating object boundaries (especially when paired with proper noise-reduction preprocessing like Median filtering), applying these manually engineered spatial features to complex medical imaging tasks reduces classification performance. Deep learning models achieve superior accuracy when provided with raw images, as they automatically optimize their own feature extraction layers.
