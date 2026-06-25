# 06. Consolidated notes: Week 4 - Aashik Arun Bobade - 11 Sep 2025

## 1. Basics of Matrices

- **Matrix multiplication**: Combine two matrices (rows × columns).
- **Transpose (Aᵀ)**: Flip rows ↔ columns.
- **Determinant**: Single value summarizing matrix (tells invertibility).
- **Eigen intuition**:

Eigenvector → direction that doesn’t change.
Eigenvalue → how much it stretches/shrinks.

---

## 2. Images as Matrices

- Grayscale image = 2D matrix (pixel intensities).
- RGB image = **3D matrix** (Height × Width × 3 color channels).
- Each pixel → [R, G, B] values (0–255).

---

## 3. Linear Algebra in Practice

- **Eigendecomposition**: Break matrix into eigenvectors + eigenvalues.
- **SVD (Singular Value Decomposition)**: General form of eigendecomp for any matrix.
- **PCA (teaser)**: Uses eigenvectors/eigenvalues of covariance matrix to reduce dimensions.
- Tools: `numpy.linalg`, `scipy.linalg`.

---

## 4. Efficient Image Operations

- **OpenCV tricks**: Image = NumPy array → easy matrix ops (resize, rotate, blur).
- **Batch transforms**: Apply same operation to many images at once.
- **GPU offload**: Libraries (like CuPy, PyTorch) use GPU for faster matrix/image operations.

---

## 5. Key Takeaways

- Matrices are the **language of images**.
- Transpose, mult., eigen concepts build up to PCA & ML.
- Efficient ops = speed (OpenCV, GPU).
- Understanding matrices = foundation for computer vision + ML.

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }