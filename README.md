# Computer Vision

Repository for Computer Vision laboratory research and experiments.

### [Lab 01: Deep Features and Classification](./lab01/)
Extracted deep features from pretrained convolutional neural networks and evaluated XGBoost classification performance on medical images. Results indicated that combining deep feature extraction with tree ensembles achieves highly accurate skin lesion classification reaching 82.93% accuracy. See the detailed results report at [results.md](./lab01/results.md).

### [Lab 02: Effect of Image Filtering on Classification](./lab02/)
Investigated the impact of spatial domain filtering on deep learning feature extraction. Evaluated ResNet101 and DenseNet121 models on skin lesion images after applying average, Gaussian, median, sharpening, and Sobel filters. Results demonstrated that manual spatial filtering degrades model accuracy. For instance, applying a Sobel filter reduced DenseNet121 accuracy from 84.75% down to 72.32%. Convolutional networks learn optimal feature extraction automatically making classical preprocessing detrimental to classification performance. See the detailed results report at [results.md](./lab02/results.md).

### [Lab 03: Edge Detection Techniques and Classification](./lab03/)
Explored first order, second order, and multi stage edge detectors including Sobel, Laplacian, and Canny algorithms. Analyzed noise sensitivity and evaluated classical machine learning and deep learning models on edge mapped images. Results showed that utilizing purely edge based representations severely reduced classification performance by discarding crucial texture and color information. For example, DenseNet121 accuracy dropped to approximately 54.8%. This confirms that convolutional networks perform best on raw input data. See the detailed results report at [results.md](./lab03/results.md).

### [Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection](./assignment01/)
Applied Canny edge detection on skin lesion images combined with different filters. The final comparison shows that filtering matters far more than the choice of edge operator. Any smoothing filter cut the estimated noise from σ ≈ 1.01 (original) to about 0.27–0.33, and the unfiltered images gave the worst overall results for both Sobel and Canny. Average and Gaussian filtering were the most effective at suppressing noise (σ ≈ 0.27), while median filtering left slightly more (σ ≈ 0.33) but was the best pairing for Canny. Overall, Average + Sobel ranked first and Gaussian + Sobel had the best boundary overlap (IoU 0.516), with Canny trailing Sobel in most pairings. Sobel's thresholded gradient magnitude gives thick, connected edges that close easily into a filled lesion region, whereas Canny's thin one-pixel edges (using low thresholds of 10–30, since the standard 50–200 range found almost no edges on smoothed dermoscopy images) leave gaps and pick up hair and texture, which lowers its IoU. These scores are measured against an Otsu-based pseudo-reference rather than expert masks, so they favour blob-like segmentation and should be read as relative rankings, not absolute accuracy. [results.md](./assignment01/results.md/)

More to come...
