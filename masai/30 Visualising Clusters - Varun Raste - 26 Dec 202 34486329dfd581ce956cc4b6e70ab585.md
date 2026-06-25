# 30. Visualising Clusters - Varun Raste - 26 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/d0601a9f-ee91-4f16-8ef9-7743f5356fd7/sOKltxKej02N2paZ.zip)

# **Lecture Note: Visualising Clusters**

---

## **1. Why Do We Need Clustering Evaluation?**

Before jumping into formulas or graphs, let’s start with a **real-life analogy**.

Imagine you’re organizing students into study groups.

You can:

- create **2 large groups** → too broad
- create **20 tiny groups** → too messy

So the big question is:

> “How many groups make sense?”
**“Are the groups actually good?”**
> 

That’s exactly why we need:

- **Elbow Method** → *How many clusters?*
- **Silhouette Score** → *How good are the clusters?*

---

## **2. K-Means Clustering (Quick Recap)**

### **What K-Means Tries to Do**

K-Means groups data such that:

- points inside a cluster are **close to each other**
- clusters are **far from each other**

### **How It Works (High Level)**

1. Choose **K** (number of clusters)
2. Assign points to nearest cluster center
3. Recalculate cluster centers
4. Repeat until stable

⚠️ **Key limitation:**

K-Means **forces you to choose K beforehand** — and that’s where Elbow & Silhouette help.

---

## **3. Elbow Method (Choosing the Right K)**

[]

[]

[]

### **Core Idea**

As K increases:

- clusters become tighter
- error (distance within clusters) decreases

But after a point, improvement slows down.

That “bend” = **elbow point**

### **What We Plot**

- **X-axis:** Number of clusters (K)
- **Y-axis:** Within-Cluster Sum of Squares (WCSS)

### **How to Explain to Students**

Think of buying phones 📱:

- Jump from ₹5k → ₹15k → big improvement
- Jump from ₹45k → ₹55k → small improvement

➡️ Stop where improvement **starts reducing sharply**

### **Interpretation**

- Sharp bend = good K
- No clear bend = data may not cluster well

### **Limitations**

❌ Subjective (different people may see different elbows)

❌ Doesn’t tell *quality* of clusters, only *compactness*

---

## **4. Silhouette Score (How Good Are the Clusters?)**

[]

[]

[]

### **Core Idea**

Silhouette score checks:

- how close a point is to its **own cluster**
- how far it is from **other clusters**

### **Score Range**

- **+1** → perfect clustering
- **0** → overlapping clusters
- **1** → wrong clustering

### **Intuitive Explanation**

Ask each data point:

> “Am I closer to my own group or someone else’s group?”
> 

### **How It’s Used**

- Try different values of K
- Compute silhouette score for each
- Choose K with **highest average score**

### **Why It’s Powerful**

✅ Measures both **cohesion** and **separation**

✅ More objective than Elbow Method

### **Limitation**

❌ Computationally expensive for large datasets

---

## **5. Combining Elbow + Silhouette (Best Practice)**

✔️ **Elbow Method** → narrow down K

✔️ **Silhouette Score** → confirm quality

👉 This combination is **industry best practice**, not using either alone.

---

## **6. DBSCAN (Density-Based Clustering)**

[]

[https://www.researchgate.net/publication/258442676/figure/fig1/AS%3A613961674272771%401523391278299/DBSCAN-core-border-and-noise-points.png]

[]

Now let’s talk about a **very different mindset**.

### **What DBSCAN Does Differently**

- Does **NOT** require K
- Groups points based on **density**
- Can find **arbitrary-shaped clusters**
- Can detect **noise/outliers**

### **Two Key Parameters**

1. **eps** → neighborhood radius
2. **min_samples** → minimum points needed to form a dense region

### **Types of Points**

- **Core points** → dense area
- **Border points** → edge of cluster
- **Noise points** → outliers

### **Why DBSCAN Is Powerful**

✅ Handles noise naturally

✅ No need to guess K

✅ Works well with irregular cluster shapes

### **Where DBSCAN Struggles**

❌ Different densities → poor performance

❌ Parameter tuning is tricky

---

## **7. K-Means vs DBSCAN (Conceptual Comparison)**

Aspect | K-Means | DBSCAN
Needs K? | Yes | No
Handles noise | ❌ No | ✅ Yes
Cluster shape | Circular | Any shape
Works with varying density | ❌ | ⚠️ Sometimes
Evaluation methods | Elbow, Silhouette | Visual / domain-based

---

## **8. When to Use What (Real-World Guidance)**

### **Use K-Means When**

- Data is clean
- Clusters are roughly similar size
- You want fast & scalable clustering

### **Use DBSCAN When**

- Outliers matter (fraud, anomaly detection)
- Cluster shapes are irregular
- You don’t know number of clusters

---

# **9. UMAP**

---

## **A. Intro (What is UMAP?)**

[Umap_space]

- **UMAP (Uniform Manifold Approximation and Projection)** is a technique that reduces high-dimensional data into **2D or 3D** so we can **see patterns clearly**.
- It helps us take very complex data (like 100+ columns) and turn it into a simple scatter plot.
- Each point in the plot represents one data record.
- Points that stay close together form **clusters**.
- UMAP is fast, works well for large datasets, and keeps the structure of data better than many older methods.

---

## **B. Why It Matters**

[UMAP]

- This UMAP scatter plot shows MNIST handwritten digits reduced into 2D space, forming **clear, well-separated clusters** for each digit (0–9).
- Each cluster is colored differently, helping you easily see how UMAP groups similar images together based on their high-dimensional features.
1. **Makes complex data simple** – converts huge datasets into easy 2D visuals.
2. **Highlights natural groups (clusters)** without forcing labels.
3. **Works faster** than many similar techniques (like t-SNE).
4. **Keeps global structure** – far apart clusters remain truly far apart.
5. **Great for exploring unknown data** before modeling.
6. **Useful in many fields** like text, images, geography, and customer behaviour.

---

## **C. Detailed Walkthrough**

1. 
**Start with high-dimensional data**
Example: A dataset with 50–100 columns (features like age, income, scores).

2. 
**Normalize/scale data**
UMAP works better when numerical values are in similar ranges.

3. 
**UMAP learns relationships**
It checks which data points are close to each other (neighbors) in high dimensions.

4. 
**Compresses the data**
UMAP reduces the dataset into 2D or 3D while trying to keep important patterns.

5. 
**Plots the result**
Now each point becomes a dot on the map.
Nearby dots = similar data items.
Far dots = very different items.

6. 
**Interpret clusters**
You look at color patterns and shapes to understand groupings.

---

## **D. Real-World Use Cases**

### **1. Customer Segmentation**

Companies use UMAP to group customers based on buying habits.

Example: Customers who always buy discounts may cluster together.

### **2. Text and NLP**

UMAP turns sentence embeddings or TF-IDF vectors into 2D plots.

You can visually inspect topics or similar sentences.

### **3. Geographic Data**

When combined with coordinates, UMAP helps map patterns in population, crime, disease, or business activity.

---

## **E. Simple Example and Explanation**

### **Code:**

```
python
import umap
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# sample dataset
data = load_digits().data

# apply UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

plt.scatter(embedding[:, 0], embedding[:, 1], s=10, cmap='Spectral')
plt.title("UMAP projection of digits dataset")
plt.show()

```

### **Explanation:**

- `load_digits()` gives us handwritten digit data (0–9), each digit stored as many features.
- UMAP reduces these features into **two numbers** for each sample → (x, y) point.
- We scatter-plot these points.
- Points representing the same digit cluster naturally (e.g., all “7”s appear close).

This shows how UMAP finds real patterns even without labels.

---

## **F. Common Confusions**

1. 
**“UMAP is a clustering method.” → No**
UMAP only reduces dimensions; it doesn’t create clusters but **reveals** them visually.

2. 
**“UMAP always gives the same plot.” → Not always**
UMAP has randomness; using `random_state=42` gives stable results.

3. 
**“Closer points always mean the same label.” → Not guaranteed**
Often true, but overlaps can happen with noisy or mixed data.

---

## **G. Key Takeaways**

1. UMAP reduces complex data into 2D or 3D for easy visualization.
2. It helps you **see natural clusters** clearly.
3. It works **fast** and scales well for big datasets.
4. It is widely used in text, images, mapping, and customer analysis.
5. UMAP does **not** label clusters — it only reveals structure.

---

# **10. GeoPandas Maps**

---

## **A. Intro (What is GeoPandas?)**

[GeoPandas_worldmaps]

- This image shows a simple GeoPandas world map created using the default **world.plot()**, useful for teaching how GeoPandas loads and visualizes geographic shapes.
- It highlights how countries are displayed as polygons on a clean, uncluttered map.
- **GeoPandas** is a Python library that helps you work with **geographic data** easily.
- It extends normal Pandas DataFrames by adding **geometry columns** (points, lines, shapes).
- You can load maps of countries, states, districts, or custom locations.
- You can plot these maps directly with simple commands.
- It helps you link your data with real locations (e.g., city points, region boundaries).

---

## **B. Why It Matters**

[geopandas_map]

- This image visualizes a world map with countries colored based on a data value, demonstrating GeoPandas' ability to combine geometry with attributes.
- It clearly shows how spatial data can be used to create meaningful, data-driven maps.
1. **Makes working with maps very easy** — no need for complicated GIS tools.
2. **Lets you combine location data with normal tables** (like Pandas).
3. **Helps visualize patterns that only make sense on a map** (e.g., population density).
4. **Supports many map formats** (.shp, GeoJSON, KML).
5. **Works with latitude/longitude directly** for plotting on world or India maps.
6. **Useful for exploratory analysis**, storytelling, and decision-making.

---

## **C. Detailed Walkthrough**

1. 
**Load geographic data**
GeoPandas has built-in datasets (like world map) or you can load your own shapefiles/GeoJSON.

2. 
**Convert your data into a GeoDataFrame**
Example: create a geometry column using latitude & longitude.

3. 
**Merge your GeoDataFrame with normal data**
Example: join a city map with population or sales data.

4. 
**Plot the map**
Use `.plot()` to draw boundaries, shapes, or points.

5. 
**Customize the visualization**
Add colors, shading, or labels to show patterns clearly.

6. 
**Interpret the map**
Look at how patterns change by location — clusters, hotspots, gaps, trends.

---

## **D. Real-World Use Cases**

### **1. City-Level Sales/Economic Analysis**

Companies map store locations, sales numbers, or delivery zones.

Helps see where performance is strong, weak, or uneven.

### **2. Public Health and Risk Mapping**

Used to map disease spread, vaccination coverage, pollution levels, or risk zones.

Makes it easy to spot high-risk and low-risk areas.

### **3. Crime or Population Density Maps**

Government and research teams map crime hotspots, population clusters, or migration patterns.

Helps make better policy decisions.

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