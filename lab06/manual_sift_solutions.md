# Manual SIFT Feature Detection and Description — Complete Solutions

**Course Code**: CS-482 / CV  
**Total Marks**: 22 Marks  
**Format**: Written Assignment Reference Solutions

---

## Question 1: Image Gradient Calculation (6 Marks)

Consider the following $4 \times 4$ grayscale image patch extracted from a $16 \times 16$ image:

### Image Patch $I$:

| Row / Col | Col 1 | Col 2 | Col 3 | Col 4 |
| :---: | :---: | :---: | :---: | :---: |
| **Row 1** | 10 | 10 | 10 | 10 |
| **Row 2** | 10 | **20** | 30 | 40 |
| **Row 3** | 10 | 30 | 50 | 70 |
| **Row 4** | 10 | 40 | 70 | 100 |

```
Matrix Representation:
[ 10   10   10   10  ]
[ 10   20   30   40  ]   <- Center pixel at (2,2) = 20
[ 10   30   50   70  ]
[ 10   40   70   100 ]
```

Using the central difference method, calculate the horizontal and vertical gradients at pixel $I(2,2) = 20$:

$$
G_x = \frac{I(x+1, y) - I(x-1, y)}{2}
$$

$$
G_y = \frac{I(x, y+1) - I(x, y-1)}{2}
$$

---

### Part a: Calculate $G_x$ and $G_y$ at pixel $I(2,2)$

Let $x$ denote horizontal index (column) and $y$ denote vertical index (row):
* Center: $I(2,2) = 20$
* Right: $I(3,2) = 30$
* Left: $I(1,2) = 10$
* Bottom: $I(2,3) = 30$
* Top: $I(2,1) = 10$

**Horizontal Gradient ($G_x$):**
$$
G_x = \frac{30 - 10}{2} = \frac{20}{2} = 10
$$

**Vertical Gradient ($G_y$):**
$$
G_y = \frac{30 - 10}{2} = \frac{20}{2} = 10
$$

---

### Part b: Calculate the gradient magnitude $M$

$$
M = \sqrt{G_x^2 + G_y^2}
$$

$$
M = \sqrt{10^2 + 10^2} = \sqrt{100 + 100} = \sqrt{200} = 10\sqrt{2} \approx 14.142
$$

---

### Part c: Calculate the gradient orientation $\theta$

$$
\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

Since $G_x = 10 > 0$ and $G_y = 10 > 0$, the vector lies in **Quadrant I**:

$$
\theta = \tan^{-1}\left(\frac{10}{10}\right) = \tan^{-1}(1) = 45^\circ \quad \left(\frac{\pi}{4} \text{ rad}\right)
$$

---

### Part d: Interpret the calculated gradient magnitude and orientation

1. **Magnitude ($M \approx 14.14$)**: Measures edge strength. The non-zero value indicates a clear intensity transition across pixel $(2,2)$ rather than a uniform region.
2. **Orientation ($\theta = 45^\circ$)**: Indicates the direction of maximum intensity increase. Pixel intensities rise diagonally toward the bottom-right, from $10$ at $(1,1)$ to $100$ at $(4,4)$.

---

## Question 2: Difference of Gaussians (DoG) (5 Marks)

A small image region at two different Gaussian scales is represented by:

### Scale $L_\sigma$:

| 10 | 20 | 10 |
|:---:|:---:|:---:|
| 20 | **80** | 20 |
| 10 | 20 | 10 |

### Scale $L_{k\sigma}$:

| 15 | 25 | 15 |
|:---:|:---:|:---:|
| 25 | **60** | 25 |
| 15 | 25 | 15 |

Calculate the Difference of Gaussians matrix:
$$
D(x, y, \sigma) = L_{k\sigma}(x, y) - L_\sigma(x, y)
$$

---

### Part a: Calculate the complete $3 \times 3$ DoG matrix

Subtract each element of $L_\sigma$ from $L_{k\sigma}$:

```
D(x, y, σ) = [ (15-10)   (25-20)   (15-10) ]   [  5    5    5 ]
             [ (25-20)   (60-80)   (25-20) ] = [  5  -20    5 ]
             [ (15-10)   (25-20)   (15-10) ]   [  5    5    5 ]
```

| 5 | 5 | 5 |
|:---:|:---:|:---:|
| 5 | **-20** | 5 |
| 5 | 5 | 5 |

---

### Part b: Identify the pixel with the largest absolute DoG value

* Boundary pixels: $|5| = 5$
* Center pixel: $|-20| = 20$

The pixel with the largest absolute DoG value is the **center pixel $(2,2)$**, with an absolute value of **$20$**.

---

### Part c: Explain whether this pixel is a potential candidate for further keypoint testing

**Yes.** SIFT identifies candidate keypoints at both local maxima and local minima of the DoG function.  
The center value ($-20$) is strictly less than all eight surrounding spatial neighbors ($+5$), forming a local minimum (a dark blob response). Therefore, it qualifies as an extremum candidate for subsequent keypoint testing.

---

### Part d: State why SIFT compares DoG values across neighboring pixels and scales before accepting a keypoint

1. **Spatial comparison (8 neighbors in current scale)**: Guarantees accurate 2D spatial localization of the feature.
2. **Scale comparison (9 neighbors in scale below, 9 in scale above)**: Ensures scale selection invariance. A keypoint is only selected if it is an extremum in both space and scale, identifying structures at their characteristic scale and providing repeatability under zoom and distance changes.

---

## Question 3: Keypoint Detection Using a $3 \times 3$ Neighborhood (5 Marks)

Given DoG values in a local neighborhood:

```
D = [  5   8   6 ]
    [  7  15   9 ]
    [  4  10   3 ]
```

| 5 | 8 | 6 |
|:---:|:---:|:---:|
| 7 | **15** | 9 |
| 4 | 10 | 3 |

The center pixel has value $15$.

---

### Part a: Identify the center pixel and its eight spatial neighbors

* **Center pixel**: $D(2,2) = 15$
* **Eight neighbors**:
  * Top-Left: $5$, Top: $8$, Top-Right: $6$
  * Left: $7$, Right: $9$
  * Bottom-Left: $4$, Bottom: $10$, Bottom-Right: $3$

---

### Part b: Determine whether the center pixel is a local maximum

Compare the center value ($15$) with all eight neighbors:
$$
15 > \{5, 8, 6, 7, 9, 4, 10, 3\}
$$
Because $15$ is strictly greater than every neighbor, the center pixel **is a local spatial maximum**.

---

### Part c: Explain why the center pixel cannot be confirmed as a final SIFT keypoint using this matrix alone

1. **Missing Scale Dimension**: SIFT requires extremum verification across 3D scale-space. The pixel must also be compared against the 9 neighbors in the scale below and 9 neighbors in the scale above (26 total neighbors).
2. **Threshold Rejection Stages**:
   * **Low-Contrast Filtering**: Candidates with $|D(\hat{x})| < 0.03$ are removed as unstable noise.
   * **Edge-Response Elimination**: Keypoints along edges are eliminated using the Hessian matrix principal curvature ratio $\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}$ with $r=10$.

---

### Part d: How many neighboring samples are compared in the full SIFT algorithm?

A total of **26 neighboring samples**:
* **8 neighbors** in the current scale ($3 \times 3$ matrix minus center)
* **9 neighbors** in the scale below
* **9 neighbors** in the scale above

$$
\text{Total} = 8 + 9 + 9 = 26 \text{ neighbors}
$$

---

## Question 4: Gradient Orientation Histogram (6 Marks)

Given four pixels with local gradients:

| Pixel | $G_x$ | $G_y$ |
| :---: | :---: | :---: |
| $P_1$ | $3$ | $4$ |
| $P_2$ | $0$ | $5$ |
| $P_3$ | $-4$ | $0$ |
| $P_4$ | $3$ | $-3$ |

---

### Part a: Calculate gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$

* **$P_1$**: $M_1 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0$
* **$P_2$**: $M_2 = \sqrt{0^2 + 5^2} = \sqrt{0 + 25} = \sqrt{25} = 5.0$
* **$P_3$**: $M_3 = \sqrt{(-4)^2 + 0^2} = \sqrt{16 + 0} = \sqrt{16} = 4.0$
* **$P_4$**: $M_4 = \sqrt{3^2 + (-3)^2} = \sqrt{9 + 9} = \sqrt{18} = 3\sqrt{2} \approx 4.243$

---

### Part b: Calculate gradient orientation in degrees (correct quadrant)

Using $\theta = \text{atan2}(G_y, G_x)$ mapped to $[0^\circ, 360^\circ)$:

* **$P_1$** ($G_x = 3 > 0, G_y = 4 > 0$): **Quadrant I**
  $$
  \theta_1 = \tan^{-1}\left(\frac{4}{3}\right) \approx 53.13^\circ
  $$

* **$P_2$** ($G_x = 0, G_y = 5 > 0$): **Positive $y$-axis**
  $$
  \theta_2 = 90.00^\circ
  $$

* **$P_3$** ($G_x = -4 < 0, G_y = 0$): **Negative $x$-axis**
  $$
  \theta_3 = 180.00^\circ
  $$

* **$P_4$** ($G_x = 3 > 0, G_y = -3 < 0$): **Quadrant IV**
  $$
  \theta_4 = 360^\circ + \tan^{-1}\left(\frac{-3}{3}\right) = 360^\circ - 45^\circ = 315.00^\circ
  $$

---

### Part c: Assign each orientation to one of eight histogram bins ($45^\circ$ each)

| Bin Index | Angular Range | Assigned Pixels |
| :---: | :---: | :---: |
| **Bin 0** | $0^\circ \le \theta < 45^\circ$ | None |
| **Bin 1** | $45^\circ \le \theta < 90^\circ$ | **$P_1$** ($\theta_1 = 53.13^\circ$) |
| **Bin 2** | $90^\circ \le \theta < 135^\circ$ | **$P_2$** ($\theta_2 = 90.00^\circ$) |
| **Bin 3** | $135^\circ \le \theta < 180^\circ$ | None |
| **Bin 4** | $180^\circ \le \theta < 225^\circ$ | **$P_3$** ($\theta_3 = 180.00^\circ$) |
| **Bin 5** | $225^\circ \le \theta < 270^\circ$ | None |
| **Bin 6** | $270^\circ \le \theta < 315^\circ$ | None |
| **Bin 7** | $315^\circ \le \theta < 360^\circ$ | **$P_4$** ($\theta_4 = 315.00^\circ$) |

---

### Part d: Construct the orientation histogram by adding gradient magnitudes

| Bin Index | Angular Range | Contributing Magnitude | Total Bin Value |
| :---: | :---: | :---: | :---: |
| **Bin 0** | $[0^\circ, 45^\circ)$ | $0$ | **$0.00$** |
| **Bin 1** | $[45^\circ, 90^\circ)$ | $M_1 = 5.0$ | **$5.00$** |
| **Bin 2** | $[90^\circ, 135^\circ)$ | $M_2 = 5.0$ | **$5.00$** |
| **Bin 3** | $[135^\circ, 180^\circ)$ | $0$ | **$0.00$** |
| **Bin 4** | $[180^\circ, 225^\circ)$ | $M_3 = 4.0$ | **$4.00$** |
| **Bin 5** | $[225^\circ, 270^\circ)$ | $0$ | **$0.00$** |
| **Bin 6** | $[270^\circ, 315^\circ)$ | $0$ | **$0.00$** |
| **Bin 7** | $[315^\circ, 360^\circ)$ | $M_4 \approx 4.24$ | **$4.24$** |

**Final 8-Bin Histogram Vector**:
$$
\mathbf{H} = [0.00, \; 5.00, \; 5.00, \; 0.00, \; 4.00, \; 0.00, \; 0.00, \; 4.24]
$$
