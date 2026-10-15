# Manual SIFT Feature Detection and Description — Complete Solutions

**Course Code**: CS-482 / CV  
**Total Marks**: 22 Marks  
**Format**: Written Assignment Reference Solutions

---

## Question 1: Image Gradient Calculation (6 Marks)

Consider the following $4 \times 4$ grayscale image patch extracted from a $16 \times 16$ image:

$$I = \begin{bmatrix}
10 & 10 & 10 & 10 \\
10 & 20 & 30 & 40 \\
10 & 30 & 50 & 70 \\
10 & 40 & 70 & 100
\end{bmatrix}$$

Using the central difference method, calculate the horizontal and vertical gradients at pixel $I(2,2) = 20$:
$$G_x = \frac{I(x+1, y) - I(x-1, y)}{2}$$
$$G_y = \frac{I(x, y+1) - I(x, y-1)}{2}$$

---

### Part a: Calculate $G_x$ and $G_y$ at pixel $I(2,2)$
Let the coordinates be defined such that $x$ represents the column index (horizontal) and $y$ represents the row index (vertical):
- Center pixel: $I(2,2) = 20$
- Right neighbor: $I(x+1, y) = I(3, 2) = 30$
- Left neighbor: $I(x-1, y) = I(1, 2) = 10$
- Bottom neighbor: $I(x, y+1) = I(2, 3) = 30$
- Top neighbor: $I(x, y-1) = I(2, 1) = 10$

**Horizontal Gradient ($G_x$):**
$$G_x = \frac{I(3, 2) - I(1, 2)}{2} = \frac{30 - 10}{2} = \frac{20}{2} = 10$$

**Vertical Gradient ($G_y$):**
$$G_y = \frac{I(2, 3) - I(2, 1)}{2} = \frac{30 - 10}{2} = \frac{20}{2} = 10$$

*(Note: In matrix row-major notation where $x$ denotes row and $y$ denotes column, the values remain identically $G_x = 10$ and $G_y = 10$.)*

---

### Part b: Calculate the gradient magnitude $M$
$$M = \sqrt{G_x^2 + G_y^2}$$
$$M = \sqrt{10^2 + 10^2} = \sqrt{100 + 100} = \sqrt{200} = 10\sqrt{2} \approx 14.142$$

---

### Part c: Calculate the gradient orientation $\theta$
$$\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)$$
Since $G_x = 10 > 0$ and $G_y = 10 > 0$, the gradient vector lies in **Quadrant I**:
$$\theta = \tan^{-1}\left(\frac{10}{10}\right) = \tan^{-1}(1) = 45^\circ \quad \left(\frac{\pi}{4} \text{ radians}\right)$$

---

### Part d: Interpret the calculated gradient magnitude and orientation
1. **Magnitude ($M \approx 14.14$)**: Represents the rate of intensity change (edge strength) at pixel $(2,2)$. A non-zero value indicates that the local neighborhood is not homogeneous and exhibits an intensity transition.
2. **Orientation ($\theta = 45^\circ$)**: Indicates the direction of greatest rate of increase in pixel intensity. Moving diagonally towards the bottom-right yields the steepest increase in brightness, matching the image patch values which increase diagonally from $10$ at $(1,1)$ to $100$ at $(4,4)$.

---

## Question 2: Difference of Gaussians (DoG) (5 Marks)

A small image region at two different Gaussian scales is represented by:
$$L_\sigma = \begin{bmatrix}
10 & 20 & 10 \\
20 & 80 & 20 \\
10 & 20 & 10
\end{bmatrix}, \qquad
L_{k\sigma} = \begin{bmatrix}
15 & 25 & 15 \\
25 & 60 & 25 \\
15 & 25 & 15
\end{bmatrix}$$

Calculate the Difference of Gaussians matrix using:
$$D(x, y, \sigma) = L_{k\sigma}(x, y) - L_\sigma(x, y)$$

---

### Part a: Calculate the complete $3 \times 3$ DoG matrix
Subtract each corresponding element of $L_\sigma$ from $L_{k\sigma}$:

$$D(x, y, \sigma) = \begin{bmatrix}
15 - 10 & 25 - 20 & 15 - 10 \\
25 - 20 & 60 - 80 & 25 - 20 \\
15 - 10 & 25 - 20 & 15 - 10
\end{bmatrix} = \begin{bmatrix}
5 & 5 & 5 \\
5 & -20 & 5 \\
5 & 5 & 5
\end{bmatrix}$$

---

### Part b: Identify the pixel with the largest absolute DoG value
- Corner and boundary pixels: $|D(x, y)| = |5| = 5$
- Center pixel $(2,2)$: $|D(2, 2)| = |-20| = 20$

The pixel with the largest absolute DoG value is the **center pixel $(2, 2)$**, with an absolute value of **$20$**.

---

### Part c: Explain whether this pixel is a potential candidate for further keypoint testing
**Yes, it is a potential candidate.**  
In SIFT, candidate keypoints are identified by detecting local extrema (both local maxima and local minima) of the Difference-of-Gaussians function.  
At the center pixel, the DoG value is $-20$, which is strictly less than all of its eight surrounding spatial neighbors (each having a value of $+5$). Because it represents a clear local minimum (a dark blob or center response surrounded by higher values), it qualifies as an extremum candidate for subsequent SIFT processing.

---

### Part d: State why SIFT compares DoG values across neighboring pixels and scales before accepting a keypoint
SIFT approximates the scale-normalized Laplacian of Gaussian (LoG) via the Difference of Gaussians (DoG). Comparing a candidate pixel against:
1. **Spatial neighbors (8 neighbors in the current scale)**: Ensures accurate 2D spatial localization of interest points.
2. **Scale neighbors (9 in the scale below and 9 in the scale above)**: Ensures scale selection invariance. A point is only accepted if its response is an extremum with respect to both space and scale. This identifies features at their characteristic scale, making the detected keypoints repeatable under zooming, camera distance changes, and image resolution variations.

---

## Question 3: Keypoint Detection Using a $3 \times 3$ Neighborhood (5 Marks)

Suppose the DoG values in a local neighborhood are:
$$D = \begin{bmatrix}
5 & 8 & 6 \\
7 & 15 & 9 \\
4 & 10 & 3
\end{bmatrix}$$
The center pixel has value 15.

---

### Part a: Identify the center pixel and its eight spatial neighbors
- **Center pixel**: $D(2, 2) = 15$
- **Eight spatial neighbors**:
  - Top-Left: $5$
  - Top: $8$
  - Top-Right: $6$
  - Left: $7$
  - Right: $9$
  - Bottom-Left: $4$
  - Bottom: $10$
  - Bottom-Right: $3$

---

### Part b: Determine whether the center pixel is a local maximum
To be a local maximum, the center pixel must be strictly greater than all eight neighbors:
$$15 > 5, \quad 15 > 8, \quad 15 > 6, \quad 15 > 7, \quad 15 > 9, \quad 15 > 4, \quad 15 > 10, \quad 15 > 3$$
Since $15$ is strictly greater than all eight spatial neighbors, the center pixel **is a local spatial maximum**.

---

### Part c: Explain why the center pixel cannot be confirmed as a final SIFT keypoint using this matrix alone
The center pixel cannot be confirmed as a final keypoint based solely on this $3 \times 3$ matrix for two reasons:
1. **Lack of Scale Comparison**: SIFT requires extremum verification in full 3D scale-space. The pixel must also be compared with the 9 samples in the adjacent scale below and the 9 samples in the adjacent scale above. It must be strictly greater (or strictly less) than all 26 surrounding neighbors.
2. **Subsequent Rejection Stages**:
   - **Low-Contrast Filtering**: Candidate keypoints with unstable contrast ($|D(\hat{x})| < 0.03$ on normalized $[0, 1]$ intensity) are discarded as noise.
   - **Edge-Response Elimination**: Keypoints situated along edges have high principal curvature along the edge but low curvature perpendicular to it (evaluated using the $2 \times 2$ Hessian matrix ratio $\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}$ with $r=10$). Edge points are discarded because they are sensitive to small amounts of noise.

---

### Part d: How many neighboring samples are compared in the full SIFT algorithm?
In the full SIFT algorithm, a candidate pixel is compared with **26 neighboring samples**:
- **8 neighbors** in the current scale ($3 \times 3$ matrix minus the center pixel)
- **9 neighbors** in the adjacent lower DoG scale ($3 \times 3$ matrix)
- **9 neighbors** in the adjacent upper DoG scale ($3 \times 3$ matrix)
$$\text{Total neighbors} = 8 + 9 + 9 = 26$$

---

## Question 4: Gradient Orientation Histogram (6 Marks)

Suppose four pixels in a local image region have the following gradient values:

| Pixel | $G_x$ | $G_y$ |
| :---: | :---: | :---: |
| $P_1$ | $3$ | $4$ |
| $P_2$ | $0$ | $5$ |
| $P_3$ | $-4$ | $0$ |
| $P_4$ | $3$ | $-3$ |

---

### Part a: Calculate gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$
- **Pixel $P_1$**:
  $$M_1 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0$$
- **Pixel $P_2$**:
  $$M_2 = \sqrt{0^2 + 5^2} = \sqrt{0 + 25} = \sqrt{25} = 5.0$$
- **Pixel $P_3$**:
  $$M_3 = \sqrt{(-4)^2 + 0^2} = \sqrt{16 + 0} = \sqrt{16} = 4.0$$
- **Pixel $P_4$**:
  $$M_4 = \sqrt{3^2 + (-3)^2} = \sqrt{9 + 9} = \sqrt{18} = 3\sqrt{2} \approx 4.243$$

---

### Part b: Calculate gradient orientation in degrees, using the correct quadrant
Using $\theta = \text{atan2}(G_y, G_x)$ mapped to the range $[0^\circ, 360^\circ)$:

- **Pixel $P_1$ ($G_x = 3 > 0, G_y = 4 > 0$)**: Lies in **Quadrant I**.
  $$\theta_1 = \tan^{-1}\left(\frac{4}{3}\right) \approx 53.13^\circ$$

- **Pixel $P_2$ ($G_x = 0, G_y = 5 > 0$)**: Lies along the **positive $y$-axis**.
  $$\theta_2 = 90.00^\circ$$

- **Pixel $P_3$ ($G_x = -4 < 0, G_y = 0$)**: Lies along the **negative $x$-axis**.
  $$\theta_3 = 180.00^\circ$$

- **Pixel $P_4$ ($G_x = 3 > 0, G_y = -3 < 0$)**: Lies in **Quadrant IV**.
  $$\theta_4 = 360^\circ + \tan^{-1}\left(\frac{-3}{3}\right) = 360^\circ - 45^\circ = 315.00^\circ$$

---

### Part c: Assign each orientation to one of eight histogram bins ($45^\circ$ each, $0^\circ$ to $360^\circ$)

Bin definitions:
- **Bin 0**: $[0^\circ, 45^\circ)$
- **Bin 1**: $[45^\circ, 90^\circ)$
- **Bin 2**: $[90^\circ, 135^\circ)$
- **Bin 3**: $[135^\circ, 180^\circ)$
- **Bin 4**: $[180^\circ, 225^\circ)$
- **Bin 5**: $[225^\circ, 270^\circ)$
- **Bin 6**: $[270^\circ, 315^\circ)$
- **Bin 7**: $[315^\circ, 360^\circ)$

**Bin Assignments**:
- $P_1$ ($\theta_1 = 53.13^\circ$): **Bin 1** ($[45^\circ, 90^\circ)$)
- $P_2$ ($\theta_2 = 90.00^\circ$): **Bin 2** ($[90^\circ, 135^\circ)$)
- $P_3$ ($\theta_3 = 180.00^\circ$): **Bin 4** ($[180^\circ, 225^\circ)$)
- $P_4$ ($\theta_4 = 315.00^\circ$): **Bin 7** ($[315^\circ, 360^\circ)$)

---

### Part d: Construct the orientation histogram by adding gradient magnitudes

Accumulate magnitudes into their respective bins:

| Bin Index | Angular Range | Contributing Pixels | Total Accumulated Magnitude |
| :---: | :---: | :---: | :---: |
| **Bin 0** | $0^\circ \le \theta < 45^\circ$ | None | **$0.00$** |
| **Bin 1** | $45^\circ \le \theta < 90^\circ$ | $P_1$ ($M_1 = 5.0$) | **$5.00$** |
| **Bin 2** | $90^\circ \le \theta < 135^\circ$ | $P_2$ ($M_2 = 5.0$) | **$5.00$** |
| **Bin 3** | $135^\circ \le \theta < 180^\circ$ | None | **$0.00$** |
| **Bin 4** | $180^\circ \le \theta < 225^\circ$ | $P_3$ ($M_3 = 4.0$) | **$4.00$** |
| **Bin 5** | $225^\circ \le \theta < 270^\circ$ | None | **$0.00$** |
| **Bin 6** | $270^\circ \le \theta < 315^\circ$ | None | **$0.00$** |
| **Bin 7** | $315^\circ \le \theta < 360^\circ$ | $P_4$ ($M_4 \approx 4.24$) | **$4.24$** |

**Final 8-Dimensional Histogram Vector**:
$$\mathbf{H} = [0.00, \; 5.00, \; 5.00, \; 0.00, \; 4.00, \; 0.00, \; 0.00, \; 4.24]$$
