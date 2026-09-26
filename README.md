# Computer Vision

Repository for Computer Vision laboratory research and experiments.

### [Lab 01: Deep Features and Classification](./lab01/)
Extracted deep features from pretrained convolutional neural networks and evaluated XGBoost classification performance on medical images. Results indicated that combining deep feature extraction with tree ensembles achieves highly accurate skin lesion classification reaching 82.93% accuracy. See the detailed results report at [results.md](./lab01/results.md).

### [Lab 02: Effect of Image Filtering on Classification](./lab02/)
Investigated the impact of spatial domain filtering on deep learning feature extraction. Evaluated ResNet101 and DenseNet121 models on skin lesion images after applying average, Gaussian, median, sharpening, and Sobel filters. Results demonstrated that manual spatial filtering degrades model accuracy. For instance, applying a Sobel filter reduced DenseNet121 accuracy from 84.75% down to 72.32%. Convolutional networks learn optimal feature extraction automatically making classical preprocessing detrimental to classification performance. See the detailed results report at [results.md](./lab02/results.md).

### [Lab 03: Edge Detection Techniques and Classification](./lab03/)
Explored first order, second order, and multi stage edge detectors including Sobel, Laplacian, and Canny algorithms. Analyzed noise sensitivity and evaluated classical machine learning and deep learning models on edge mapped images. Results showed that utilizing purely edge based representations severely reduced classification performance by discarding crucial texture and color information. For example, DenseNet121 accuracy dropped to approximately 54.8%. This confirms that convolutional networks perform best on raw input data. See the detailed results report at [results.md](./lab03/results.md).

More to come...
