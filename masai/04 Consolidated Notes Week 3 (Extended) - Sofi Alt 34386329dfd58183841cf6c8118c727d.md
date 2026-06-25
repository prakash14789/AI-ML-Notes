# 04. Consolidated Notes: Week 3 (Extended) - Sofi Altamsh - 16 Aug 2025

## 1. Scalars vs Vectors

- **Scalar** → single value (`5`, `3.14`, `"Alice"`).
- **Vector** → sequence of values (1D NumPy array).
- Useful in AI/ML: store & process many numbers at once (features, images, datasets).

---

## 2. Python Lists vs NumPy Arrays

Feature | List | NumPy Array
Types | Mixed | Single dtype
Math Ops | Concatenation | Element-wise math
Speed | Slow | Fast (C-based)
Functions | Loops | Vectorized ops

✅ NumPy arrays are **faster & better for math**.

---

## 3. Vector Arithmetic

- **Add/Subtract** → element-wise (`u+v`, `u-v`).
- **Dot Product** → multiply then sum (`np.dot(a,b)`).
- **Cross Product (3D)** → perpendicular vector (`np.cross(a,b)`).
- **Element-wise Multiply** → Hadamard product (`a*b`).
- **Scalar Multiply** → multiply every element by constant.

---

## 4. Broadcasting

- NumPy can **stretch smaller arrays** for operations.
- Rule: dimensions align from **right**, size `1` expands.
- Example: `(2,3)` + `(3,)` → works.
- ❌ Fails if mismatched & no `1`.

---

## 5. Fancy Indexing

- Select multiple elements using index arrays.
- **1D**: `arr[[1,3,5]] → [20,40,60]`.
- **2D**: `mat[[0,2],[1,0]] → [2,7]`.
- Methods:

Select rows: `mat[[0,2], :]`
Select cols: `mat[:, [1,3]]`
Sub-matrix: `mat[np.ix_([0,2],[1,3])]`
- **Important**: Fancy indexing returns a **copy**, not a view.

---

## 6. Modifying Arrays

- Assign single/multiple values via indices.
- Example:
`arr[[1,3]] = 99
mat[[0,1,2],[0,1,2]] = 0   # set diagonal to 0`

---

## 7. Views vs Copies

- **Slicing** → View (changes reflect in original).
- **Fancy indexing** → Copy (changes don’t affect original).

---

## 8. Vectorization

- Write code for **whole arrays at once**, no loops.
- Benefits:

⚡ Much faster (10–1000x).
✨ Cleaner, shorter code.
🖥 Better use of CPU.
- Example:
`arr1 + arr2   # vectorized
[arr1[i]+arr2[i] for i in range(n)]  # slow loop`

---

## 9. Vectorized Operations

- Arithmetic: `a+b`, `a*b`, `a-b`.
- Logical: `scores >= 60` → Boolean mask.
- Aggregations: `np.mean(arr)`, `np.max(arr)`, `np.cumsum(arr)`.

---

## 10. Ufuncs (Universal Functions)

- Fast, element-wise math functions.
- Examples:
`np.sqrt(arr), np.exp(arr), np.sin(arr), np.maximum(arr,2)`

---

## 11. Performance

- NumPy vectorization is **100x–1000x faster** than loops.
- Measure with:

`%timeit` (Jupyter)
`timeit` module
- Check memory with `sys.getsizeof(arr)`.

---

## 12. Real-World Uses

- Data analysis (filter rows).
- ML preprocessing (feature selection).
- Image processing (pixel extraction).
- Scientific computing (matrix ops).

---

## ✅ Quick Summary Table

Concept | Code | Takeaway
Scalar | x=5 | Single value
Vector | np.array([1,2]) | Sequence of values
Add | a+b | Element-wise sum
Dot | np.dot(a,b) | Sum of products
Cross | np.cross(a,b) | Perpendicular vector
Broadcasting | A+(3,) | Auto-shape math
Fancy Indexing | arr[[1,3]] | Select elements
Ufunc | np.exp(arr) | Fast element-wise
View vs Copy | slicevsfancy | Memory handling
Vectorization | arr1+arr2 | Much faster than loops

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