# Lab 06: Landmark Recognition and Localization Using the SIFT Algorithm

## 1. Introduction
This report details the implementation and evaluation of an automated landmark recognition and localization pipeline using the Scale-Invariant Feature Transform (SIFT) algorithm. The system matches query photographs against a reference database of five landmarks using Brute-Force Matcher with Lowe's ratio test, applies RANSAC homography estimation for geometric localization, and tests robustness under rotation, scale change, illumination variation, viewpoint differences, and non-database queries.

## 2. Experimental Setup
*   **Reference Database**: 5 landmarks (Eiffel Tower, Taj Mahal, Great Wall of China, Statue of Liberty, Colosseum) with multiple reference viewpoints.
*   **Feature Extraction**: SIFT (128-dimensional descriptors, contrast threshold 0.03, edge threshold 10, Gaussian $\sigma = 1.6$).
*   **Matcher**: Brute-Force Matcher with Euclidean ($L_2$) distance.
*   **Ambiguity Filtering**: Lowe's ratio test ($k=2$, ratio threshold $\tau = 0.75$).
*   **Localization**: RANSAC Homography ($cv2.findHomography$, reprojection threshold 5.0, inlier threshold $\ge 10$, convex polygon validation).
*   **Decision Logic**: Minimum match count threshold ($\ge 15$) coupled with RANSAC geometric verification.

## 3. Results and Observations

### Table 1: Performance Evaluation Across Test Conditions
| Test Condition | True Landmark | Predicted Landmark | Keypoints | Reliable Matches | RANSAC Inliers | Localized | Decision | Time (ms) |
|---|---|---|---|---|---|---|---|---|
| Original Landmark | Eiffel Tower | Eiffel Tower | 208 | 208 | 208 | Yes | RECOGNIZED | 151.3 |
| Rotated ($45^\circ$) | Eiffel Tower | Eiffel Tower | 285 | 99 | 85 | Yes | RECOGNIZED | 158.6 |
| Resized ($50\%$ Scale) | Eiffel Tower | Eiffel Tower | 65 | 50 | 41 | Yes | RECOGNIZED | 30.0 |
| Reduced Brightness ($0.4\times$) | Eiffel Tower | Eiffel Tower | 204 | 192 | 192 | Yes | RECOGNIZED | 143.9 |
| Different Viewpoint | Eiffel Tower | Eiffel Tower | 208 | 208 | 208 | Yes | RECOGNIZED | 130.3 |
| Unknown Landmark | None (Unknown) | Statue of Liberty | 83 | 34 | 22 | No | REJECTED* | 114.7 |

*\*Note: For the unknown landmark query, raw descriptor matches exceeded the minimum count threshold, but RANSAC geometric verification rejected localization due to degenerate, non-convex projection geometry.*

### Key Observations
*   **Scale and Illumination Invariance**: SIFT maintained high inlier counts under $50\%$ downscaling ($41$ inliers) and $0.4\times$ intensity reduction ($192$ inliers), verifying that normalized local gradient representations remain invariant to monotonic brightness shifts and scale changes.
*   **Rotation Invariance**: A $45^\circ$ image rotation retained $85$ RANSAC inliers and successfully localized the landmark, confirming the stability of canonical orientation assignment from gradient histograms.
*   **Role of Geometric Verification**: Relying exclusively on match count is insufficient; ambiguous landmarks can produce false positive correspondences. RANSAC homography estimation verifies planar geometric consistency, rejecting degenerate projections.

## 4. Discussion & Conclusion
SIFT extracts distinctive, scale- and rotation-invariant local features by detecting extrema in Difference-of-Gaussians scale-space and assigning canonical orientations based on local gradient histograms. In industrial and tourism applications, coupling SIFT descriptors with Lowe's ratio test and RANSAC homography estimation provides essential geometric verification, filtering out false correspondences and enabling precise object localization.
