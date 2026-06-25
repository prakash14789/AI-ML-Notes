# 07. Consolidated notes: Week 6 - Aashik Arun Bobade - 13 Sep 2025

## 1. Pandas Essentials

- **Create DataFrame**:
`import pandas as pd
df = pd.DataFrame({'Name': ['A', 'B'], 'Score': [90, 85]})`
- **Indexing**: Access by row/col → `df.loc[0, 'Name']` or `df.iloc[0,1]`.
- **Cleaning**:

Drop NA → `df.dropna()`
Fill NA → `df.fillna(0)`
Rename → `df.rename(columns={'Score':'Marks'})`
- **Joins**:
`pd.merge(df1, df2, on='id', how='left')`

---

## 2. Basic Plots with Pandas

- **Line plot**: `df.plot(x='Date', y='Sales', kind='line')` → trends over time.
- **Bar plot**: `df.plot(x='Product', y='Profit', kind='bar')` → compare categories.
- **Scatter plot**: `df.plot(x='Age', y='Income', kind='scatter')` → relationships.
- **Design principles**: Use titles, axis labels, clean legends → `plt.xlabel("X Axis")`.

---

## 3. Data Wrangling Clinic

- **Messy CSVs**:
`df = pd.read_csv('file.csv', header=None)  # no headers
df.columns = ['Name','Age','Salary']`
- **Merges**: Inner vs Outer joins. Example:
`pd.merge(customers, orders, on='cust_id', how='inner')`
- **Tidy data rules**:

Each **column = variable**
Each **row = observation**
Each **cell = value**
- **Time-series tricks**:
`df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date').resample('M').mean()`

---

## 4. Matplotlib Mastery

- **Custom styles**:
`import matplotlib.pyplot as plt
plt.style.use('seaborn')`
- **Subplots**:
`fig, ax = plt.subplots(1,2)
ax[0].plot(df['x'], df['y'])
ax[1].bar(df['x'], df['y'])`
- **Annotations**:
`plt.annotate("Peak", xy=(2020, 500), xytext=(2021, 520),
             arrowprops=dict(facecolor='red'))`
- **Export quality**: `plt.savefig('plot.png', dpi=300, bbox_inches='tight')`

---

## 5. Key Takeaways

- Pandas = **data backbone** → clean, transform, merge datasets.
- Visualization (line, bar, scatter) = quick insights.
- Tidy data → easier analysis + modeling.
- Matplotlib = **customization + publication-quality** figures.

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