# 28. Lecture notes - K-Means & DBSCAN - Dr. Surya Prakash - 22 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/6642dc17-0cc8-47c9-b651-2fdd85c47b55/10jzIjaFzkmUAK1F.zip)

# **K-Means & DBSCAN — Lecture Notes**

## **Prerequisites**

Before starting this lesson, you should be comfortable with:

1. **Basic Python** — variables, loops, lists.
2. **NumPy & Pandas basics** — working with arrays and DataFrames.
3. **Meaning of “features”** — simple numeric properties describing data.
4. **Basic idea of clustering** — grouping similar items together.
5. **Matplotlib/Seaborn basics** — making simple plots.
6. **Simple math concepts** — distance between points.

---

## **1. What You’ll Learn**

By the end of this lesson, you will learn to:

1. **Understand distance metrics** and why clustering depends on them.
2. **Explain K-Means** step-by-step in a simple way.
3. **Use the Elbow Method** to choose the best number of clusters.
4. **Use the Silhouette Score** to check cluster quality.
5. **Understand DBSCAN** and why it works well for uneven shapes.
6. **Apply density clustering** for tasks like geo-segmentation.

---

# **2. Detailed Explanation**

---

# **2.1 Distance Metrics**

---

## A. Intro — What Are Distance Metrics?

[Distance_metrics]

- Distance metrics are simple formulas that tell us **how far two data points are from each other**.
- Clustering algorithms like **K-Means and DBSCAN** rely heavily on distance to decide which points belong together.

---

## **B. Why It Matters**

1. **Clustering depends fully on distance** — groups are formed by placing similar (close) points together.
2. **Wrong distance = wrong clusters** — the whole model can fail if we choose the wrong metric.
3. **Different data needs different distances** — numeric, binary, text, and spatial data require different formulas.
4. **Impacts cluster shape** — some metrics create round clusters, others create stretched/irregular groups.
5. **Important for scaling** — distance gets affected if features are not normalized.
6. **DBSCAN uses distance to detect density** — distance defines which points count as “neighbors.”

---

## **C. Detailed Walkthrough**

### **1. Euclidean Distance (Most Common)**

- Measures the straight-line distance between two points.
- Great for continuous numeric features.
- Makes clusters circular in shape.

Formula:

d=(x1−x2)2+(y1−y2)2d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}

d=(x1−x2)2+(y1−y2)2

---

### **2. Manhattan Distance**

- Measures distance by adding absolute differences.
- Good for grid-like paths (e.g., city blocks).
- Creates diamond-shaped clusters.

Formula:

d=∣x1−x2∣+∣y1−y2∣d = |x_1 - x_2| + |y_1 - y_2|

d=∣x1−x2∣+∣y1−y2∣

---

### **3. Cosine Distance**

- Focuses on angle between points, not magnitude.
- Useful when size doesn’t matter (text vectors, customer preference ratios).

---

### **4. Haversine Distance**

- Measures distance on the Earth’s surface (lat-long).
- Very useful for geo-segmentation and map clustering.

---

### **5. Minkowski Distance**

- A generalized form of Euclidean and Manhattan.
- Gives flexibility in how “strictly” we measure distance.

---

## **D. Real-World Use Cases**

### **1. Customer Segmentation**

Retailers compare customers using Euclidean or Manhattan distance to group shoppers with similar spending patterns, visit frequency, or purchase behavior.

### **2. Document/Text Similarity**

Search engines use cosine distance to compare documents and group similar articles or detect duplicate content.

### **3. Geo-Clustering (Location-Based Apps)**

Delivery apps like Zomato/Swiggy use Haversine distance to group customers and restaurants geographically to optimize delivery zones.

---

## **E. Simple Examples**

### **Example 1: Euclidean Distance Between Two Points**

```
python
import numpy as np

p1 = np.array([2, 3])
p2 = np.array([7, 9])

dist = np.linalg.norm(p1 - p2)
print(dist)

```

**Explanation:**

- We calculate straight-line distance.
- NumPy `norm` handles the formula automatically.
- Useful for K-Means cluster assignment.

---

## **F. Common Confusions**

1. 
**“Euclidean is always best” — No.**
Works only when data is properly scaled and clusters are round.

2. 
**“Distance is unaffected by units” — Wrong.**
Features like age and income must be normalized first.

3. 
**“Cosine distance measures how far points are” — Incorrect.**
It measures **angle**, not magnitude or true spatial distance.

---

## **G. Key Takeaways**

1. Distance tells clustering algorithms who is “close” vs “far.”
2. Choice of metric drastically changes cluster shape and results.
3. Normalize data before computing distance for reliable output.
4. Cosine is best for text or preference data; Haversine for maps.
5. K-Means relies heavily on Euclidean distance; DBSCAN adapts to several metrics.

---

# **2.2 Elbow Method**

---

[Elbow_Method]

## A. Intro — What Is the Elbow Method?

- The Elbow Method is a simple technique used to **find the best number of clusters (K)** for K-Means.
- It checks how the **within-cluster error** decreases as we increase the number of clusters and identifies the point where improvement slows down — the “elbow.”

---

## **B. Why It Matters**

1. **Prevents random guessing of cluster count** — gives a structured way to pick K.
2. **Avoids underfitting/overfitting** — too few clusters oversimplify; too many overfit noise.
3. **Improves model stability** — choosing the right K makes clusters more meaningful.
4. **Reduces computation time** — fewer clusters = faster K-Means.
5. **Provides visual clarity** — the elbow graph is easy to interpret even for beginners.
6. **Works for many domains** — customers, products, text, behavior patterns.

---

## **C. Detailed Walkthrough**

### **1. Compute WCSS (Within-Cluster Sum of Squares)**

- For each possible value of K (e.g., 1 to 10), run K-Means.
- Calculate **how much error** each cluster has.

### **2. Plot WCSS vs. K**

- X-axis → Number of clusters (K).
- Y-axis → WCSS (how tightly grouped the clusters are).

### **3. Identify the “Elbow”**

- The point where **WCSS stops decreasing sharply**.
- Before elbow → big improvement.
- After elbow → very small improvement (not worth adding more clusters).

### **4. Choose That K**

- This K usually produces the best balance between simplicity and accuracy.

---

## **D. Real-World Use Cases**

### **1. Customer Segmentation**

Companies use the Elbow Method to decide how many customer groups exist naturally (e.g., budget vs. premium vs. occasional buyers).

### **2. Marketing Campaign Targeting**

Helps marketers create optimal clusters of users so that ads and recommendations can be personalized without creating too many tiny groups.

### **3. Product Categorization**

E-commerce sites use elbow analysis to find the right number of product clusters (e.g., price-based, style-based, brand-based patterns).

---

## **E. Simple Examples**

### **Example 1: Basic Elbow Plot Using K-Means**

```
python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(data)  # 'data' is your numeric dataset
    wcss.append(km.inertia_)

plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

```

**Explanation:**

- Runs K-Means for K = 1 to 9.
- Stores the WCSS values.
- Plots the curve so you can visually find the elbow.

---

### **Example 2: Automatically Detecting the Elbow (Simple Logic)**

```
python
import numpy as np

diffs = np.diff(wcss)
second_diffs = np.diff(diffs)

elbow_k = np.argmin(second_diffs) + 2
print("Elbow K =", elbow_k)

```

**Explanation:**

- First difference = how much WCSS drops.
- Second difference = drop in improvement.
- The minimum second difference often corresponds to elbow.

---

## **F. Common Confusions**

1. 
**“The elbow must be obvious” — Not true.**
Sometimes the curve is smooth and the elbow is subtle.

2. 
**“Elbow works for all clustering” — Wrong.**
It mainly applies to **K-Means**, not DBSCAN.

3. 
**“Lowest WCSS is best” — Incorrect.**
K=1 always gives the lowest WCSS; the goal is balance, not minimum.

---

## **G. Key Takeaways**

1. Elbow Method helps determine the best cluster count (K).
2. Looks at how WCSS decreases as K increases.
3. Choose the point where improvement slows — the “elbow.”
4. Prevents overfitting clusters or underfitting patterns.
5. Works great with K-Means and numeric datasets.

---

# **2.3 Silhouette**

[Silhouette]

## A. Intro — What Is the Silhouette Method?

The **silhouette method** checks how well each point fits inside its assigned cluster.

- “Each bar represents how well a point fits within its cluster. Wider bars (yellow/green) indicate points are **well matched** to their cluster, while narrower bars show **weaker clustering**.”

It compares:

- How close a point is to its own cluster
- How far it is from other clusters

This gives a **0–1 score per point** showing “how correct” the clustering feels.

---

## **B. Why It Matters**

1. **Checks cluster quality** – helps you see if your clusters actually make sense.
2. **Prevents wrong K choices** – warns you when you selected too many or too few clusters.
3. **Detects overlapping clusters** – shows when clusters are not well-separated.
4. **Highlights outliers** – points with very low silhouette usually don’t belong anywhere.
5. **Works across algorithms** – compatible with K-Means, DBSCAN, and others.
6. **Gives visual feedback** – silhouette plots clearly show cluster strengths/weaknesses.

---

## **C. Detailed Walkthrough**

1. 
**Compute average distance to own cluster (a):**
For each point, calculate how close it is to other points in the same group.
Lower value = better fit.

2. 
**Compute average distance to nearest other cluster (b):**
Find how far the point is from the closest outside cluster.
Higher value = better separation.

3. 
**Apply silhouette formula:**s=max(a,b)b−a
s=b−amax⁡(a,b)s = \frac{b - a}{\max(a, b)}
This normalizes the value between **-1 and +1**.

4. 
**Interpret the score:**

**+1** → perfect fit
**0** → on boundary
**1** → assigned to wrong cluster

5. 
**Cluster-wise evaluation:**
We average all point scores in a cluster to see which cluster is strong or weak.

6. 
**Overall silhouette value:**
Average the entire dataset’s silhouette values → final clustering quality score.

---

## **D. Real-World Use Cases**

1. 
**Customer Segmentation Validation:**
After clustering customers based on spending or behavior, silhouette helps verify if groups are meaningful or overlapping.

2. 
**Image Segmentation Quality Check:**
Used to confirm whether pixel groups (segments) are well-separated or blended, improving accuracy for medical or satellite imaging.

3. 
**Location/Geo Segmentation:**
For city zones or delivery points, silhouette ensures clusters represent real geographic patterns instead of random grouping.

---

## **E. Simple Examples**

### **Example 1 — Clean Separation**

Suppose we cluster shopping customers into **"low spenders"**, **"medium spenders"**, **"high spenders"**.

If the silhouette value for each cluster is **0.70+**, it means:

- Clusters are distinct
- Points are closer inside their cluster
- Spending levels clearly differ
This confirms the segmentation works well.

---

### **Example 2 — Overlapping Clusters**

Imagine clustering cities based on temperature + humidity.

If silhouette value is **0.15**, it suggests:

- Cities in hot/humid regions overlap with moderate regions
- Clustering is not clear
- Better features or different cluster count may be needed

This helps you realize the clustering is weak and needs improvement.

---

## **F. Common Confusions**

1. 
**“Silhouette = Silhouette Score” confusion**
No — **silhouette is the method**, **silhouette score is the value** produced by the method.

2. 
**Thinking a high score guarantees perfect clustering**
Not always. It only checks separation, not if the features used were actually meaningful.

3. 
**Assuming it works for any data shape**
Silhouette works best when clusters are somewhat separated; for strange shapes, DBSCAN works better.

---

## **G. Key Takeaways**

1. Silhouette measures how well each point fits inside its cluster.
2. It compares closeness to own cluster vs other clusters.
3. Scores range from **1 to +1**, higher is better.
4. Helps you detect wrong cluster assignments and overlaps.
5. Useful across multiple algorithms, not only K-Means.

---

# **2.4 Density Clustering (DBSCAN)**

[DBSCAN]

## A. Intro — What Is Density Clustering?

Density clustering groups points based on how “crowded” an area is.

Instead of forcing a fixed number of clusters (like K-Means), DBSCAN finds clusters that naturally form based on dense regions — and labels low-density points as **noise/outliers**.

---

## **B. Why It Matters**

1. 
**No need to choose K**
DBSCAN automatically detects how many clusters exist.

2. 
**Finds irregular shapes**
Works well when clusters are curved or not perfectly round (unlike K-Means).

3. 
**Handles noise naturally**
Points in sparse areas become “outliers” instead of being force-fit.

4. 
**Works well for real-world messy data**
Perfect for spatial, geographic, and unevenly distributed datasets.

5. 
**Good performance for large datasets**
Efficient even when thousands of points exist.

6. 
**Better for clusters of different sizes**
Doesn’t assume all clusters should be similar.

---

## **C. Detailed Walkthrough**

[DBSCAN_Clustering]

1. **Set two parameters:**

**eps (ε):** How close points must be to be considered neighbors.
**min_samples:** Minimum number of neighbors to form a dense region.

2. **Classify each point as:**

**Core point:** Has enough neighbors → part of a cluster.
**Border point:** Near a core point but not dense enough.
**Noise point:** Neither core nor border → outlier.

3. **Start forming clusters:**

Pick an unvisited point.
If it’s a **core point**, create a new cluster and expand outward.

4. **Expand cluster:**

Recursively add all reachable points (density-connected points).
Continue until cluster cannot expand.

5. **Move to next unvisited point:**

Repeat until all points are labeled.

6. **Return final results:**

Cluster labels
Outlier/noise detection
Cluster shapes and sizes

---

## **D. Real-World Use Cases**

1. 
**GPS/Geo-Segmentation (Location Groups):**
DBSCAN naturally finds dense location hubs like restaurants, accident hotspots, delivery zones — even when shapes are irregular.

2. 
**Anomaly Detection in Transactions:**
Sparse or unusual behavior (like fraud patterns) appears as noise points, making DBSCAN ideal for financial anomaly detection.

3. 
**Image Pixel Grouping (Shape Detection):**
Clusters can follow complex shapes in pixel intensity data — useful in medical imaging, satellite segmentation, etc.

---

## **E. Simple Examples**

### **Example 1 — Finding Densely Packed Locations**

Imagine a city where coffee shops form clusters in busy areas and isolated ones in suburbs.

DBSCAN will:

- Group shops in crowded downtown
- Mark isolated suburban shops as noise
This reveals natural city “business clusters”.

---

### **Example 2 — Identifying Outliers in User Activity**

If most users log in multiple times daily but a few log in once a month, DBSCAN will:

- Cluster frequent users
- Mark rare logins as noise
This helps detect inactive or suspicious accounts.

---

## **F. Common Confusions**

1. 
**“DBSCAN is just K-Means with noise handling”**
No — DBSCAN is density-based and does NOT force round shapes or a fixed number of clusters.

2. 
**“eps should be large for better clusters”**
Bigger eps may merge clusters incorrectly; tuning is required.

3. 
**“All noise points are mistakes”**
Noise often represents natural outliers — not errors in clustering.

---

## **G. Key Takeaways**

1. DBSCAN clusters based on density, not distance to a center.
2. It automatically detects cluster count — no K required.
3. Works great for irregular shapes and uneven distributions.
4. Naturally identifies outliers as noise.
5. Perfect for spatial, behavioral, and real-world messy data.

---

# **2.5 Geo-Segmentation**

---

[Geo_Segmentation]

- This image shows data points grouped into segments on a map using colors. It helps visualize how customers or regions are clustered geographically for better analysis.

## **A. Intro — What Is Geo-Segmentation?**

Geo-segmentation is the process of grouping customers, regions, or data points based on their **geographic location**.

It uses clustering techniques (like K-Means or DBSCAN) along with **latitude–longitude coordinates** to discover spatial patterns.

---

## **B. Why It Matters**

1. 
**Improves targeted marketing**
Companies can identify which regions respond better to specific campaigns and customize offers.

2. 
**Helps in resource allocation**
Businesses decide where to open stores, increase delivery fleets, or allocate field staff.

3. 
**Reveals hidden geographic patterns**
Clusters show real-world behavior like high-demand zones, tourist hotspots, or low-sales regions.

4. 
**Useful for logistics and delivery optimization**
Clustering locations reduces travel cost and helps plan efficient delivery routes.

5. 
**Supports public policy & city planning**
Governments use geo-segmentation to identify high-pollution areas, healthcare gaps, traffic congestion, etc.

6. 
**Enhanced decision-making using spatial intelligence**
Geo-clustering transforms latitude-longitude data into actionable business insights.

---

## **C. Detailed Walkthrough**

1. 
**Collect geographic data**
Extract latitude, longitude, city, region, pin codes, or geohashes.

2. 
**Choose a clustering algorithm**

Use **K-Means** when regions have spherical boundaries.
Use **DBSCAN** when regions have irregular shapes and density variations.

3. 
**Scale coordinates (optional)**
Some models need lat-long standardization to avoid biased clustering.

4. 
**Run clustering on geographic points**
The model groups nearby points into clusters.

5. 
**Visualize clusters on a map**
Use tools like Folium, Plotly, or QGIS to see real spatial patterns.

6. 
**Interpret & apply insights**
Example: Cluster A = high-value customers; Cluster B = low delivery coverage; Cluster C = high churn region.

---

## **D. Real-World Use Cases**

### **1. Food Delivery Zonal Optimization**

Platforms like Swiggy or Zomato cluster delivery addresses to create service zones.

This reduces delivery time and balances rider workload.

### **2. Retail Store Expansion**

Retail chains cluster cities based on customer density and purchasing power.

This helps them decide **where to open new branches** to maximize profit.

### **3. Telecom Network Planning**

Telecom companies analyze signal complaints, user density, and tower distribution.

Clusters highlight **poor network regions** requiring new towers.

---

## **E. Simple Good Examples**

### **Example 1: Customer Location Clustering**

A company has 10,000 customer addresses with lat-long values.

They apply **K-Means** to form 5 regions:

- Region 1 → High purchases
- Region 2 → Low engagement
- Region 3 → Frequent returns
**Explanation:**
K-Means helps divide the market geographically so the company can assign regional managers and design specific marketing plans.

---

### **Example 2: Delivery Route Optimization Using DBSCAN**

A courier company clusters all delivery points in Bangalore.

DBSCAN forms clusters of dense areas (e.g., Koramangala, Whitefield) and marks isolated far-away points as “noise.”

**Explanation:**

This helps the company send dedicated delivery teams to dense clusters while planning special delivery for isolated locations.

---

## **F. Common Confusions**

1. 
**Geo-segmentation ≠ demographic segmentation**
Geo looks at location; demographic looks at age, gender, income.

2. 
**Latitude-longitude are not linear distances**
1-degree difference is not the same everywhere; you often need Haversine distance.

3. 
**K-Means may not work well for irregular shapes**
Many real-world regions are uneven, making **DBSCAN** or **HDBSCAN** more suitable.

---

## **G. Key Takeaways**

1. Geo-segmentation clusters customers or locations using geographic data.
2. Useful for marketing, planning, logistics, city management, and resource allocation.
3. K-Means is good for simple, spherical regions; DBSCAN handles complex spatial shapes.
4. Visualization on maps is crucial for understanding real spatial clusters.
5. Geo-segmentation turns raw location data into **actionable business insights**.

---

# **3. Mental Models**

### **1. “Clustering is Grouping by Similarity” Mental Model**

Always think of clustering as grouping things that are *closer* to each other and *farther* from others—distance is the heart of all clustering logic.

### **2. “Right Number of Clusters = Balance Between Fit & Simplicity”**

Use the Elbow and Silhouette methods to avoid too many or too few clusters; imagine finding a “sweet spot” where clusters are meaningful without being forced.

### **3. “Shape of Data Decides the Algorithm”**

If clusters are round → use K-Means.

If clusters are irregular/noisy → use DBSCAN.

Always choose based on *data shape*, not preference.

### **4. “Density Reveals Hidden Patterns”**

Think of DBSCAN as a map of crowded vs. empty spaces. Crowded = cluster. Empty = noise. This mental model helps you understand non-linear clusters.

### **5. “Location Patterns Reflect Human Behavior”**

Geo-segmentation tells you that people behave in patterns based on where they live, travel, or move—clustering + coordinates = location intelligence.

---

# **4. Key Takeaways**

### **1. Distance metrics decide how similarity is measured.**

Euclidean, Manhattan, Cosine, and Haversine determine how “close” two points are depending on data type and problem.

### **2. Elbow and Silhouette help you choose the best number of clusters.**

Elbow shows cost vs. simplicity; Silhouette shows cluster quality and separation.

### **3. K-Means works best for spherical, evenly sized clusters.**

Fast, simple, but sensitive to noise, scale, and irregular shapes.

### **4. DBSCAN excels at density-based, irregular, noisy data.**

Finds clusters of any shape and naturally detects outliers without specifying k.

### **5. Geo-segmentation is clustering applied to real-world geography.**

Useful for marketing, transport, planning, and location-based personalization.

### **6. No single clustering algorithm works for all datasets.**

Choose based on shape, noise, scale, and distribution of your data.

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