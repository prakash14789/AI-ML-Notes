# 08. Editorial: Mid Trimester Evaluation (Attempt 2) - Sofi Altamsh - 8 Oct 2025

**1. Your teammate reports “My code runs locally but fails in CI with import errors.” Which practice most directly prevents this?**

a) Add the project directory to `.gitignore`.

b) Use a virtual environment (or container) and pin dependencies in a lock file.

c) Increase logging verbosity.

d) Use print debugging only.

**Answer:** b

**Explanation:** Using virtual environments and pinned dependencies ensures everyone has the exact packages and versions, preventing CI import errors.

---

**2. In NumPy, what does the expression `np.arange(3) + np.array([10])` produce?**

a) `[10, 11, 12]`

b) `[3, 3, 3]`

c) `[0, 1, 2, 10]`

d) Error: cannot add arrays of different shapes

**Answer:** a

**Explanation:** `np.arange(3)` gives `[0,1,2]`; adding `[10]` broadcasts the scalar 10 across all elements → `[10,11,12]`.

---

**3. Which computes the elementwise square of a NumPy array `a`?**

a) `np.dot(a, a)`

b) `np.vectorize(lambda x: x**2)(a)`

c) `a ** 2`

d) `np.matmul(a, a)`

**Answer:** c

**Explanation:** `a ** 2` squares each element individually. `np.dot` and `np.matmul` are for matrix multiplication.

---

**4. If `x = np.arange(12).reshape(3,4)`, what is `x.T.shape`?**

a) `(12,)`

b) `(3, 4)`

c) `(3,)`

d) `(4, 3)`

**Answer:** d

**Explanation:** `.T` transposes the matrix → rows and columns swap → shape `(4,3)`.

---

**5. When an image loaded with OpenCV appears with swapped colors in Matplotlib, what's the cause?**

a) Matplotlib cannot display JPEG images.

b) OpenCV reads images as BGR while Matplotlib expects RGB.

c) OpenCV automatically converts to grayscale.

d) Matplotlib expects images as float in [0,1] only.

**Answer:** b

**Explanation:** OpenCV loads images in BGR format; Matplotlib assumes RGB → colors look swapped. Conversion fixes this.

---

**6. Which call creates a Figure and a 2×2 grid of Axes you can index as `axes[0,1]`?**

a) `plt.figure()`

b) `plt.subplot(2, 2)`

c) `fig, axes = plt.subplots(2, 2)`

d) `plt.create_subplotgrid(2,2)`

**Answer:** c

**Explanation:** `plt.subplots(2,2)` returns both figure and a 2×2 array of Axes objects.

---

**7. If `A.shape == (5,3)` and `B.shape == (3,4)`, which is a valid matrix multiplication?**

a) `B @ A`

b) `A @ B`

c) `A * B` (elementwise)

d) `np.multiply(A, B)`

**Answer:** b

**Explanation:** For `A @ B`, inner dimensions must match (3 in A and 3 in B) → result `(5,4)`.

---

**8. Which is the recommended way to convert a DataFrame column `col` to datetimes?**

a) `df['col'] = df['col'].astype('object')`

b) `df['col'] = pd.to_datetime(df['col'])`

c) `df['col'] = df['col'].apply(str)`

d) `df['col'] = df['col'].map(lambda x: x)`

**Answer:** b

**Explanation:** `pd.to_datetime()` safely converts strings or numbers to datetime dtype.

---

**9. Which SymPy call computes the derivative of `expr` with respect to `x`?**

a) `sp.diff(expr, x)`

b) `expr.gradient(x)`

c) `expr.derive()`

d) `sp.derivative(expr)`

**Answer:** a

**Explanation:** `sp.diff(expr, x)` computes symbolic derivatives in SymPy.

---

**10. Which Git command fetches remote changes and merges them into your current branch?**

a) `git fetch`

b) `git pull`

c) `git clone`

d) `git remote`

**Answer:** b

**Explanation:** `git pull` = `fetch` + `merge`; updates your current branch with remote commits.

---

**11. Which NumPy function returns indices that would sort an array?**

a) `np.sort()`

b) `np.argmax()`

c) `np.argsort()`

d) `np.argpartition()`

**Answer:** c

**Explanation:** `np.argsort()` returns indices that would sort the array.

---

**12. What does `df.drop_duplicates(subset=['id'], keep='last')` do?**

a) Removes the `id` column.

b) Keeps the first occurrence of each `id`.

c) Drops duplicate rows based on `id`, keeping the last occurrence.

d) Drops rows where `id` is NaN only.

**Answer:** c

**Explanation:** `keep='last'` preserves the last row for each duplicate `id` and drops others.

---

**13. To select rows where the first column is zero from a NumPy 2D array `arr`, which is most efficient?**

a) `for`-loop checking each row in Python

b) `arr[arr[:, 0] == 0]`

c) Convert to list of lists and filter

d) Use `np.where(arr == 0)` without slicing

**Answer:** b

**Explanation:** Boolean indexing selects rows efficiently without Python loops.

---

**14. Which of the following operations on a NumPy `ndarray` *returns a view* (i.e., does not make a copy)?**

a) `arr[1:4]`

b) `arr + 0`

c) `np.concatenate([arr, arr])`

d) `arr.copy()`

**Answer:** a

**Explanation:** Slicing returns a view; other operations create new arrays.

---

**15. What will be the output after these calls?**

```python
def f(a, L=[]):
    L.append(a)
    return L

f(1)
f(2)
print(f(3))

```

a) `[3]`

b) `[1, 2, 3]`

c) `[1]`

d) Error

**Answer:** b

**Explanation:** Default list `L` persists across calls → `[1,2,3]`.

---

**16. Which dtype change most reduces memory usage while usually preserving acceptable precision for ML workloads?**

a) Use `np.float64` instead of `np.float32`

b) Use `np.float32` instead of `np.float64`

c) Use `np.object` dtype for numeric arrays

d) Use `np.int64` instead of `np.int32`

**Answer:** b

**Explanation:** `float32` uses half the memory of `float64` and usually suffices for ML.

---

**17. Which is a method to fill missing values in column `col` with the column mean?**

a) `df['col'] = df['col'].dropna()`

b) `df['col'].fillna(df['col'].mean(), inplace=True)`

c) `df.replace('', df['col'].mean())`

d) `df.fillna(0)`

**Answer:** b

**Explanation:** `fillna()` with mean fills missing values; `inplace=True` modifies the column directly.

---

**18. Which NumPy function returns the unique sorted values of an array?**

a) `np.distinct()`

b) `np.unique()`

c) `np.nunique()`

d) `np.uniq()`

**Answer:** b

**Explanation:** `np.unique()` returns sorted unique values of an array.

---

**19. What is the result of `(1,2) + (3,)` in Python?**

a) `(1, 2, 3)`

b) `[1, 2, 3]`

c) `(4,)`

d) Error

**Answer:** a

**Explanation:** Tuples concatenate → `(1,2) + (3,) = (1,2,3)`.

---

**20. Which of these are valid ways to convert an OpenCV BGR image `img` to RGB?**

a) `img[:, :, ::-1]`

b) `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

c) `cv2.toRGB(img)`

d) Both a and b

**Answer:** d

**Explanation:** Both slicing and `cv2.cvtColor` correctly convert BGR → RGB. `cv2.toRGB` doesn’t exist.

---

**21. What is the result of adding these arrays in NumPy?**

```python
a = np.array([1, 2, 3])         # shape (3,)
b = np.array([[10], [20]])     # shape (2,1)
a + b

```

a) `array([11, 12, 13, 21, 22, 23])`

b) `array([[11, 12, 13], [21, 22, 23]])`

c) `array([[10, 20, 1], [2, 3, 0]])`

d) Raises `ValueError` due to shape mismatch

**Answer:** b

**Explanation:** Broadcasting expands shapes → (2,1) + (3,) → (2,3).

---

**22. What does `np.broadcast_to(x, (4,3))` do?**

a) Return a view of `x` broadcast to shape `(4,3)` where possible, without copying.

b) Always allocate and copy `x` tiled to `(4,3)`.

c) Reshape `x` in-place to `(4,3)`.

d) Raise an error unless `x` already has shape `(4,3)`.

**Answer:** a

**Explanation:** `np.broadcast_to` broadcasts array without copying memory (view).

---

**23. You have a DataFrame `df` containing a datetime column `'ts'` (spanning multiple years) and a numeric column `'value'`. You want a Series of monthly means of `'value'` indexed by the first day of each month, and you want months with no data to appear (as `NaN`). Which is the best pandas expression?**

a) `df.groupby(df['ts'].dt.month)['value'].mean()`

b) `df.set_index('ts').resample('MS')['value'].mean()`

c) `df.groupby(pd.Grouper(key='ts', freq='M'))['value'].mean()`

d) `df.groupby(df['ts'].dt.to_period('M'))['value'].mean()`

**Answer:** b

**Explanation:** `resample('MS')` ensures the Series is indexed by month start and includes missing months as NaN.

---

**24. For arrays larger than available RAM, which NumPy feature helps you treat on-disk data as arrays?**

a) `np.memmap()`

b) `np.bigarray()`

c) `np.diskarray()`

d) `np.swapaxes()`

**Answer:** a

**Explanation:** `np.memmap()` maps a file on disk as an array → efficient for very large arrays.

---

**25. In Python, what is the output of:**

```python
x = [1, 2, 3]
y = x
y.append(4)
print(x)

```

a) `[1, 2, 3]`

b) `[1, 2, 3, 4]`

c) `[4]`

d) Error

**Answer:** b

**Explanation:** `y` points to the same list object as `x`; modifying `y` also modifies `x`.

---

**26. Which function splits a string `s` into words?**

a) `s.split()`

b) `s.join()`

c) `s.strip()`

d) `s.partition()`

**Answer:** a

**Explanation:** `split()` divides a string into a list of words by whitespace by default.

---

**27. You want to replace all NaNs in a DataFrame `df` with zeros. Which works?**

a) `df.fillna(0, inplace=True)`

b) `df.replace(np.nan, 0)`

c) `df.dropna()`

d) `df.astype(int)`

**Answer:** a

**Explanation:** `fillna()` replaces NaNs; `inplace=True` modifies `df` directly.

---

**28. Which Matplotlib function sets the x-axis label?**

a) `plt.xlabel('X')`

b) `plt.set_xlabel('X')`

c) `plt.label('X')`

d) `plt.axis('X')`

**Answer:** a

**Explanation:** `xlabel()` sets the label for the x-axis.

---

**29. Which is true about `df.loc` vs `df.iloc` in pandas?**

a) `.loc` uses integer positions, `.iloc` uses labels.

b) `.loc` uses labels, `.iloc` uses integer positions.

c) Both use only labels.

d) Both use only integer positions.

**Answer:** b

**Explanation:** `.loc` selects rows/columns by label; `.iloc` by integer index.

---

**30. Which of the following is correct to compute the cumulative sum of a NumPy array `arr`?**

a) `np.sum(arr, axis=0)`

b) `np.cumsum(arr)`

c) `arr.cumsum(axis=1)`

d) Both b and c

**Answer:** d

**Explanation:** `np.cumsum()` or `arr.cumsum()` computes cumulative sum; axis can be specified for 2D arrays.

---

**31. In pandas, which operation drops rows with any missing value?**

a) `df.dropna(axis=1)`

b) `df.dropna(how='any')`

c) `df.fillna(0)`

d) `df.replace(np.nan, 0)`

**Answer:** b

**Explanation:** `dropna(how='any')` removes rows with any NaN.

---

**32. You want to flatten a 2D NumPy array `a` to 1D. Which is correct?**

a) `a.reshape(-1)`

b) `a.flatten()`

c) `a.ravel()`

d) All of the above

**Answer:** d

**Explanation:** All three methods convert a 2D array to 1D; `ravel()` returns a view if possible, `flatten()` returns a copy.

---

**33. What does the Pandas command `df['col'].value_counts(normalize=True)` do?**

a) Counts unique values in `col`

b) Counts unique values and converts counts to proportions

c) Replaces missing values in `col`

d) Sorts `col` in ascending order

**Answer:** b

**Explanation:** `normalize=True` returns the proportion of each unique value instead of counts.

---

**34. Which NumPy method randomly shuffles elements in-place along the first axis?**

a) `np.shuffle(arr)`

b) `np.random.shuffle(arr)`

c) `arr.shuffle()`

d) `np.random.permutation(arr)`

**Answer:** b

**Explanation:** `np.random.shuffle()` shuffles an array along the first axis in-place. `permutation()` returns a new array.

---

**35. Which of the following avoids the Python “for loop” for elementwise operations on arrays?**

a) List comprehension

b) NumPy vectorization

c) Pandas `apply()`

d) All of the above

**Answer:** d

**Explanation:** Vectorized operations are faster than Python loops; `apply()` or comprehensions can be alternatives.

---

**36. Which is true for the Python `is` operator?**

a) Compares values

b) Compares object identity (memory location)

c) Always same as `==`

d) None of the above

**Answer:** b

**Explanation:** `is` checks if two variables point to the same object.

---

**37. What is the result of `np.dot(a, b)` if `a.shape = (2,3)` and `b.shape = (3,1)`?**

a) `(2,1)` array

b) `(3,2)` array

c) Scalar

d) Error

**Answer:** a

**Explanation:** Dot product inner dimensions must match → result shape `(2,1)`.

---

**38. Which of the following commands resets the index of a DataFrame `df` and drops the old index?**

a) `df.reset_index()`

b) `df.reset_index(drop=True)`

c) `df.set_index(0)`

d) `df.drop_index()`

**Answer:** b

**Explanation:** `drop=True` ensures old index is not added as a column.

---

**39. Which Python feature allows functions to have a variable number of positional arguments?**

a) `*args`

b) `**kwargs`

c) `lambda`

d) `yield`

**Answer:** a

**Explanation:** `*args` collects extra positional arguments; `**kwargs` collects extra keyword arguments.

---

**40. Which method in pandas returns the first N rows of a DataFrame?**

a) `df.first(N)`

b) `df.head(N)`

c) `df.top(N)`

d) `df.sample(N)`

**Answer:** b

**Explanation:** `df.head()` returns first N rows.

---

**41. Which of the following avoids copying data when slicing a NumPy array?**

a) `arr[1:5]`

b) `arr.copy()[1:5]`

c) `np.concatenate([arr, arr])`

d) `arr + 0`

**Answer:** a

**Explanation:** Slicing returns a view; others create copies.

---

**42. You want to compute the row-wise sum of a 2D NumPy array `a`. Which is correct?**

a) `a.sum(axis=0)`

b) `a.sum(axis=1)`

c) `np.cumsum(a, axis=1)`

d) `np.dot(a, a.T)`

**Answer:** b

**Explanation:** `axis=1` sums across columns → row-wise sum.

---

**43. Which of the following selects rows in a DataFrame where column `col` is not null?**

a) `df[df.col.notnull()]`

b) `df[df.col.isnull()]`

c) `df.dropna(col)`

d) `df.fillna(0)`

**Answer:** a

**Explanation:** `.notnull()` returns a boolean mask for non-NaN values.

---

**44. In matplotlib, which style is used to create a scatter plot?**

a) `plt.plot(x, y)`

b) `plt.scatter(x, y)`

c) `plt.bar(x, y)`

d) `plt.line(x, y)`

**Answer:** b

**Explanation:** `plt.scatter()` is specifically for scatter plots.

---

**45. Which Pandas method combines two DataFrames vertically (stacking rows)?**

a) `pd.concat([df1, df2], axis=0)`

b) `pd.merge(df1, df2)`

c) `df1.append(df2, axis=1)`

d) `df1.join(df2)`

**Answer:** a

**Explanation:** `concat` with `axis=0` stacks rows; `merge` is for joining on keys.

---

# **MSQs (1–16) with Explanations**

---

**1. Which of the following are valid ways to select a column `'col'` in pandas?**

a) `df['col']`

b) `df.col`

c) `df.iloc[:, 'col']`

d) `df.loc[:, 'col']`

**Answer:** a, b, d

**Explanation:** `.iloc` uses integer index → `'col'` is invalid. Other methods work.

---

**2. Which operations can broadcast a scalar over a NumPy array `a`?**

a) `a + 5`

b) `a * 2`

c) `np.dot(a, 2)`

d) `a ** 3`

**Answer:** a, b, d

**Explanation:** Scalar arithmetic broadcasts; `np.dot(a,2)` is invalid (dot requires arrays).

---

**3. Which of the following creates a 2×3 NumPy array of zeros?**

a) `np.zeros((2,3))`

b) `np.zeros([2,3])`

c) `np.zeros(2,3)`

d) `np.array([[0]*3]*2)`

**Answer:** a, b, d

**Explanation:** All valid except `np.zeros(2,3)` → TypeError (tuple needed).

---

**4. Which commands return the number of rows in a DataFrame `df`?**

a) `len(df)`

b) `df.shape[0]`

c) `df.count()`

d) `df.size`

**Answer:** a, b

**Explanation:** `len(df)` and `df.shape[0]` → number of rows; `df.count()` counts non-NaN per column; `df.size` = total elements.

---

**5. Which methods remove NaN values from a DataFrame?**

a) `df.dropna()`

b) `df.fillna(0)`

c) `df.replace(np.nan, 0)`

d) `df.isnull()`

**Answer:** a

**Explanation:** Only `dropna()` removes rows/columns; others replace or detect NaNs.

---

**6. Which NumPy functions return a new array rather than modifying in-place?**

a) `np.add(a, b)`

b) `a += b`

c) `np.multiply(a, b)`

d) `a *= b`

**Answer:** a, c

**Explanation:** `np.add` and `np.multiply` return new arrays; `+=` modifies `a` in-place.

---

**7. Which are valid ways to iterate over DataFrame rows?**

a) `for idx, row in df.iterrows():`

b) `for row in df.itertuples():`

c) `for i in range(len(df)):`

d) `for row in df:`

**Answer:** a, b, c

**Explanation:** Iterating directly `for row in df:` loops over column names, not rows.

---

**8. Which of the following are mutable in Python?**

a) list

b) tuple

c) dict

d) set

**Answer:** a, c, d

**Explanation:** Tuples are immutable; lists, dicts, sets are mutable.

---

**9. Which statements are true about `.loc` and `.iloc` in pandas?**

a) `.loc` can use boolean masks

b) `.iloc` accepts slice objects

c) `.iloc` uses column names

d) `.loc` can select rows and columns simultaneously

**Answer:** a, b, d

**Explanation:** `.iloc` uses integer indices, not column names.

---

**10. Which commands return unique values in a pandas column `col`?**

a) `df['col'].unique()`

b) `df['col'].drop_duplicates()`

c) `df['col'].nunique()`

d) `df['col'].value_counts()`

**Answer:** a, b

**Explanation:** `.nunique()` counts, `.value_counts()` returns counts, not unique values.

---

**11. Which methods can convert a pandas column to lowercase strings?**

a) `df['col'].str.lower()`

b) `df['col'].apply(str.lower)`

c) `df['col'].map(str.lower)`

d) `df['col'].lower()`

**Answer:** a, b, c

**Explanation:** `.lower()` alone on Series won’t work; string accessor or apply/map needed.

---

**12. Which methods fill missing values in pandas?**

a) `df['col'].fillna(0)`

b) `df.fillna(method='ffill')`

c) `df.dropna()`

d) `df.replace(np.nan, 0)`

**Answer:** a, b, d

**Explanation:** `dropna()` removes rows, doesn’t fill NaNs.

---

**13. Which are valid ways to select rows based on conditions?**

a) `df[df['col'] > 0]`

b) `df.loc[df['col'] > 0]`

c) `df.query('col > 0')`

d) `df.iloc[df['col'] > 0]`

**Answer:** a, b, c

**Explanation:** `.iloc` requires integer positions → invalid with boolean mask.

---

**14. Which of the following are valid ways to reset index?**

a) `df.reset_index()`

b) `df.reset_index(drop=True)`

c) `df.set_index('col')`

d) `df.index = range(len(df))`

**Answer:** a, b, d

**Explanation:** `.set_index()` changes index to a column, not reset.

---

**15. Which of the following pandas operations preserve original DataFrame unless `inplace=True`?**

a) `df.drop(columns=['col'])`

b) `df.fillna(0)`

c) `df.rename(columns={'old':'new'})`

d) `df.dropna()`

**Answer:** a, b, c, d

**Explanation:** All methods return a new DataFrame unless `inplace=True`.

---

**16. Which are valid ways to access elements of a NumPy 2D array `a`?**

a) `a[0,1]`

b) `a[0][1]`

c) `a[0, 1:3]`

d) `a[1:2, 0:2]`

**Answer:** a, b, c, d

**Explanation:** All forms are valid indexing/slicing.

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