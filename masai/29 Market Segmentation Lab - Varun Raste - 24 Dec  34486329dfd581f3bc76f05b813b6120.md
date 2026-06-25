# 29. Market Segmentation Lab - Varun Raste - 24 Dec 2025

## [In-Class Resources](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/97e046aa-61c1-4ad3-a65a-2fee7dc5617a/FZ0dBOvUYtwrqxyY.zip)

# **Market Segmentation Lab — Lecture Notes**

## **Prerequisites**

1. Basic Python (variables, lists, functions).
2. How to load and view datasets using Pandas.
3. Very basic idea of what “clustering” means.
4. How to install and import scikit-learn.
5. Simple plotting using Matplotlib or Seaborn.

---

## **1. What You’ll Learn**

**By the end of this lesson, you will be able to:**

1. Use **scikit-learn clustering algorithms** (like K-Means) for segmentation.
2. Choose the **right scaler** (StandardScaler / MinMaxScaler) based on your data.
3. Understand why scaling affects clustering results.
4. Create **customer segments** from raw data.
5. Build simple **personas** for each cluster.
6. Interpret segmentation results in a practical, business-friendly way.

---

# **2. Detailed Explanation**

---

# **2.1. sklearn cluster**

## **A. Intro — What Is sklearn cluster?**

[Sklearn_Cluster]

- Sklearn cluster is a module in scikit-learn that provides ready-to-use clustering algorithms like **K-Means**, **DBSCAN**, and **Agglomerative Clustering**, allowing you to group similar data points without labels.
- This image shows how K-Means groups similar data points into clear clusters based on distance.
- The colored regions help you visually understand how sklearn.cluster separates customers into meaningful segments.

---

## **B. Why It Matters**

1. **Makes clustering easy** — You don’t need to write complex math; the functions do it for you.
2. **Helps discover patterns** — You can find groups in your data that you didn’t know existed.
3. **Works for many types of data** — Numbers, locations, customer info, and more.
4. **Industry standard** — Used by companies for segmentation, anomaly detection, and recommendations.
5. **Fast and efficient** — Algorithms are optimized, so even large datasets work smoothly.
6. **Beginner-friendly** — Only a few lines of code are enough to start clustering.

---

## **C. Detailed Walkthrough**

### **1. Import the algorithm**

```
python
from sklearn.cluster import KMeans

```

### **2. Prepare the data**

You select the features you want to cluster, such as:

```
python
X = data[['age', 'income']]

```

### **3. Create the model**

```
python
model = KMeans(n_clusters=3)

```

### **4. Fit the model**

This step learns the patterns in your data:

```
python
model.fit(X)

```

### **5. Get cluster labels**

Each row gets assigned to a group:

```
python
data['cluster'] = model.labels_

```

### **6. Visualize or analyze**

Plot the clusters or study the differences.

---

## **D. Real-World Use Cases**

### **1. Customer Segmentation**

Businesses use clustering to find groups like “high spenders,” “budget buyers,” or “frequent visitors.” This helps in targeted marketing and personalized offers.

### **2. Image Compression**

Clustering reduces image colors by grouping similar pixel shades. This makes images smaller while keeping them visually acceptable.

### **3. Fraud or Anomaly Detection**

If a data point doesn’t fit into any cluster properly, it may indicate unusual activity — useful in banking, cybersecurity, and transaction monitoring.

---

## **E. Examples**

### **Example 1 — Simple K-Means Clustering**

```
python
from sklearn.cluster import KMeans
import pandas as pd

data = pd.DataFrame({
    'age': [22, 25, 45, 52, 23],
    'income': [20, 25, 60, 80, 22]
})

model = KMeans(n_clusters=2)
model.fit(data)

data['cluster'] = model.labels_

```

**Explanation:**

- We create a small table of age and income.
- K-Means finds 2 groups.
- The model labels each row as cluster 0 or 1.

---

### **Example 2 — Clustering Points on a Map**

```
python
from sklearn.cluster import DBSCAN

coords = [[10, 20], [11, 19], [50, 80], [52, 82]]
model = DBSCAN(eps=5, min_samples=2)
labels = model.fit_predict(coords)

```

**Explanation:**

- Points close to each other become part of a cluster.
- Points far away become noise (label = -1).
- Useful for geo-segmentation.

---

## **F. Common Confusions**

1. **“K-Means works for all data”** — No, it struggles with noisy or unevenly shaped clusters.
2. **“Scaling doesn’t matter”** — Wrong; clustering changes completely if data is not scaled.
3. **“More clusters = better results”** — Too many clusters can overfit and confuse the analysis.

---

## **G. Key Takeaways**

1. `sklearn.cluster` gives you multiple clustering algorithms to choose from.
2. K-Means is best for simple, round-shaped clusters.
3. DBSCAN works well when clusters are irregular or noisy.
4. Scaling the data is extremely important for good results.
5. Clustering helps discover natural groups without any labels.

---

# **2.2 Scaler Choice**

---

## A. Intro — What Is Scaler Choice?

[Scaling]

- This image shows how scaling transforms features so they have similar ranges. Without scaling, one large feature can dominate distance-based algorithms like K-Means.
- When we use machine learning algorithms like **K-Means**, **DBSCAN**, or **Hierarchical Clustering**, the numbers in each column (feature) may have very different ranges.

Example:

- `total_bill` can be **2 to 50**
- `tip` may be **1 to 10**
- `size` may be **1 to 6**

Because of this, the algorithm may give more importance to large numbers and ignore small ones.

**Scaling** fixes this by bringing all values to a similar scale.

---

## **B. Why It Matters**

1. **Avoids Bias** – Features with large numbers (like ₹, weight, income) won’t dominate smaller ones.
2. **Accurate Clusters** – Distance-based algorithms give better separation when all features are in the same range.
3. **Faster Convergence** – Algorithms like K-Means run faster and more smoothly when data is scaled.
4. **Better Interpretability** – Easier to compare features when they share a similar scale.
5. **Essential for Most Models** – Any model that uses distances or gradients needs scaling.
6. **Prevents Wrong Results** – Without scaling, clusters may look random or misleading.

---

## **C. Detailed Walkthrough**

[Feature_Scaling]

### **1. When do we scale?**

- If the algorithm calculates **distance** (Euclidean, Manhattan)
- If the algorithm is sensitive to **feature ranges**
- If one column has numbers much larger than others

### **2. Common Scalers**

### **a) StandardScaler**

- Converts data so the **mean becomes 0**
- And the **standard deviation becomes 1**
- Good for: Normal distribution data, K-Means, PCA

### **b) MinMaxScaler**

- Converts values to a range of **0 to 1**
- Best when you want to **preserve shape** of data
- Works well for clustering and neural networks

### **c) RobustScaler**

- Uses **median** instead of mean
- Best when data has **outliers** (extreme values)

### **3. Steps to Apply**

1. Select the columns you want to scale
2. Choose the right scaler
3. Fit the scaler on the training data
4. Transform the data
5. Use the scaled data for clustering or modelling

---

## **D. Real-World Use Cases**

### **1. Customer Spending Analysis**

If a dataset contains:

- Age (18–70)
- Monthly spending (1,000–200,000)
- Number of visits (1–20)

Spending will dominate the clustering. Scaling ensures all variables contribute fairly.

### **2. Marketing Segmentation**

Features like:

- Website time
- Number of pages visited
- Purchase value

All have different ranges—scaling helps get cleaner market segments.

### **3. Delivery/Logistics Clustering**

Attributes like:

- Distance travelled
- Delivery time
- Weight
- Cost

These vary widely. Scaling gives more accurate delivery zone clusters.

---

## **E. Example**

### **Example Code**

```
python
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.DataFrame({
    'total_bill': [20, 35, 50],
    'tip': [3, 5, 10],
    'size': [2, 4, 6]
})

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled, columns=df.columns)

```

### **Explanation :**

- We load three columns
- We apply **StandardScaler**
- It converts values so they are on a similar scale
- Now no column will dominate the others
- You can safely use this scaled data for clustering like K-Means

---

## **F. Common Confusions**

1. 
**“Should I scale all models?”**
No. Tree-based models (Random Forest, XGBoost) don’t need scaling.

2. 
**“Do I scale before train-test split?”**
No.
You **fit on train** and **transform both train and test**.

3. 
**“Which scaler is the best?”**
There is **no single best** — it depends on the dataset and method.

---

## **G. Key Takeaways**

1. Scaling is essential when features have different ranges.
2. K-Means and DBSCAN work best with scaled data.
3. StandardScaler = mean 0, SD 1 → best default choice.
4. MinMaxScaler = 0 to 1 → best when you want to keep shape.
5. Always scale **after** splitting the data and **before** clustering.

---

# **2.3 Persona Crafting**

---

## A. Intro — What Is Persona Crafting?

Persona crafting means creating a **simple, imaginary profile** of a customer group based on data.

Instead of looking at raw numbers, we turn clusters into **human-like descriptions** such as:

- “Budget-friendly diners”
- “High spenders”
- “Weekend family visitors”

Personas help marketers, product teams, and businesses understand **who** their customers are and **what they need**.

---

## **B. Why It Matters**

1. 
**Makes Data Understandable**
Instead of confusing numbers, teams get a clear picture of each customer type.

2. 
**Improves Decision-Making**
Helps decide pricing, marketing messages, and promotions for each segment.

3. 
**Better Targeting**
Businesses can communicate differently to each persona, increasing conversions.

4. 
**Builds Customer-Centric Products**
Helps design better offers, features, or services based on real behaviour.

5. 
**Gives Direction to Strategies**
Teams know **what customers value**, so planning becomes easier.

6. 
**Useful Across Teams**
Sales, marketing, design, and analytics all use personas to stay aligned.

---

## **C. Detailed Walkthrough**

### **1. Start with Clusters**

After running K-Means or DBSCAN, we get groups like:

- Cluster 0
- Cluster 1
- Cluster 2

These clusters are just numbers—they don’t mean anything yet.

### **2. Study Each Segment’s Features**

For each cluster, check:

- Average spending
- Average tip
- Visit frequency
- Group size
- Day/time preferences
- Demographic patterns (if available)

### **3. Identify Key Behaviour**

Ask questions like:

- Do they spend more or less than average?
- Do they come in groups or alone?
- Do they visit on weekends or weekdays?
- Are they price-sensitive?

### **4. Give the Persona a Name**

Names make it easy to remember.

Examples:

- “The Regular Saver”
- “The Luxury Spender”
- “The Family Group”

### **5. Write a Short Description**

Explain:

- Who they are
- What they like
- Their goals
- Their pain points
- Behaviour pattern

### **6. Use Visual Aids (Optional)**

Include:

- A small icon
- Bullet points
- Key stats

This helps others quickly understand the persona.

---

## **D. Real World Use Cases**

### **1. Restaurant Marketing**

Restaurants use personas like:

- “Solo office-goers”
- “Weekend families”
- “Date-night couples”

This helps run targeted offers (e.g., weekday lunch combo for office workers).

### **2. E-commerce Personalization**

Online stores segment users into:

- “Bargain hunters”
- “Premium buyers”
- “Festival shoppers”

This helps decide:

- Which product to promote
- What discount to show
- When to send emails

### **3. Banking & Finance**

Banks create personas like:

- “First-time salary earners”
- “High net-worth individuals”
- “Senior citizens needing stability”

This helps design:

- Credit cards
- Investment plans
- Loan packages

---

## **E. Example and Explanation**

### **Example Cluster Output**

Suppose clustering gives this:

Cluster | Avg Bill | Avg Tip | Size | Visit Pattern
0 | ₹18 | ₹2 | 2 | Weekdays
1 | ₹45 | ₹7 | 3 | Weekends
2 | ₹10 | ₹1 | 1 | Random

### **Converted Personas**

### **Persona 1: “Budget Solo Diners” (Cluster 2)**

- Spend very less
- Come alone
- Quick visits
- Price-sensitive

### **Persona 2: “Family Weekend Spenders” (Cluster 1)**

- Higher spending
- Visit in groups
- Prefer weekends
- Value comfort & experience

### **Persona 3: “Regular Office Visitors” (Cluster 0)**

- Moderate spenders
- Come during weekdays
- Steady behaviour

### Explanation

- Raw numbers become easy-to-understand human profiles.
- These help businesses create personalized marketing and better customer strategies.

---

## **F. Common Confusions**

1. 
**“Are personas real customers?”**
No. They are **imaginary characters** based on real data patterns.

2. 
**“Do personas replace clusters?”**
No.
Clusters = Machine output
Personas = Human-friendly interpretation of clusters

3. 
**“Do personas stay the same forever?”**
No.
Customer behaviour changes → personas must be updated regularly.

---

## **G. Key Takeaways**

1. Personas convert data segments into human-like customer profiles.
2. They make clustering results understandable to non-technical teams.
3. Good personas include behaviours, preferences, goals, and pain points.
4. Personas help in marketing, product planning, and customer experience.
5. Personas must be updated as customer behaviour changes over time.

---

# **3. Mental Models**

### **1. "Clustering = Grouping by Similarity"**

- Think of clustering like arranging similar items in a supermarket—milk with dairy, chips with snacks. Algorithms simply do this mathematically.

### **2. "Scaling = Making Features Speak the Same Language"**

- Different features have different scales (₹, %, km). Scaling ensures no feature dominates the others. It’s like adjusting microphone volumes so all speakers sound balanced.

### **3. "Personas = Human Summaries of Clusters"**

- A persona converts raw cluster numbers into human-friendly stories like **“Budget Eaters”**, **“Luxury Diners”**, or **“Frequent Visitors”**.

---

# **4. Final Key Takeaways**

### **1. Clustering helps find hidden patterns**

- You can group customers even when no labels exist, enabling powerful insights.

### **2. Scaling improves clustering quality**

- StandardScaler, MinMaxScaler, and RobustScaler can dramatically change cluster shapes and results.

### **3. Persona crafting bridges data → business**

- Clusters alone are technical; personas convert them into meaningful profiles decision-makers can use.

### **4. sklearn provides easy-to-use clustering tools**

- With a few lines of code (`KMeans`, `DBSCAN`, scaling), you can run complete segmentation pipelines.

### **5. Market segmentation becomes actionable only when clusters + scaling + personas are combined**

- All three steps form one workflow: **preprocess → cluster → interpret**.

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