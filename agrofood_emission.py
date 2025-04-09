import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os

# Optional: Print out files from the folder
for dirname, _, filenames in os.walk('D://agro_food_emission//.venv//agrofoodemission.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load dataset
df = pd.read_csv('./.venv/agrofoodemission.csv')

print("Dataset Overview:")
print(df.head(), "\n")

print("Dataset Info:")
print(df.info(), "\n")

print("Descriptive Statistics:")
print(df.describe().T, "\n")

# --- Visualizations ---

# 1. Pairplot on a sample of numeric features
numeric_df = df.select_dtypes(include=[np.number])
sns.pairplot(numeric_df.sample(n=500, random_state=41))  # Reduced sample size
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".1f", cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# 3. Line plot: Total emission over years
plt.figure(figsize=(8, 4))
df.groupby('Year')['total_emission'].sum().plot(marker='o')
plt.title("Total Emission Over Years")
plt.ylabel("Total Emission")
plt.xlabel("Year")
plt.show()

# 4. Violin plot for specific years
plt.style.use('default')
sns.set(style='whitegrid')
x1 = df[df['Year'] == 1990]['total_emission']
x2 = df[df['Year'] == 2000]['total_emission']
x3 = df[df['Year'] == 2010]['total_emission']
x4 = df[df['Year'] == 2020]['total_emission']
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)
ax.violinplot([x1, x2, x3, x4])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['1990', '2000', '2010', '2020'])
ax.set_xlabel('Year')
ax.set_ylabel('Total Emission')
ax.set_title('Violin Plot of Emissions by Year')
plt.show()

# 5. Bar plots - Top & bottom 20 countries by total emission
top_20 = df.groupby('Area')['total_emission'].sum().sort_values(ascending=False).head(20)
bottom_20 = df.groupby('Area')['total_emission'].sum().sort_values(ascending=True).head(20)

plt.figure(figsize=(10, 6))
top_20.plot(kind='bar', color='green')
plt.title("Top 20 Countries by Total Emission")
plt.ylabel("Total Emission")
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(10, 6))
bottom_20.plot(kind='bar', color='red')
plt.title("Bottom 20 Countries by Total Emission")
plt.ylabel("Total Emission")
plt.xticks(rotation=45, ha='right')
plt.show()

# 6. Lineplot visualization for Total Emission for Russian Federation and China
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[df['Area'].isin(['Russian Federation', 'China'])], x='Year', y='total_emission', hue='Area')
plt.title('Total Emission Over Time')
plt.xlabel('Year')
plt.ylabel('Total Emission')
plt.legend(title='Country')
plt.grid(True)
plt.show()

# Lineplot visualization for Average Temperature °C for Russian Federation and China
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[df['Area'].isin(['Russian Federation', 'China'])], x='Year', y='Average Temperature °C', hue='Area')
plt.title('Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend(title='Country')
plt.grid(True)
plt.show()

# Scatterplot: Total emission vs Average Temperature °C (2019)
df_2019 = df[df['Year'] == 2019]
plt.figure(figsize=(12, 8))
plt.axvline(x=0, color='red', lw=1, ls='--', alpha=0.8)
plt.axhline(y=df_2019['Average Temperature °C'].median(), color='red', lw=1, ls='--', alpha=0.8)
sns.scatterplot(data=df_2019, x='total_emission', y='Average Temperature °C')
plt.title('Total Emission vs Average Temperature (2019)')
plt.grid(True)
plt.show()

# Scatterplot: Total emission <= 0 vs Average Temperature °C (2019)
plt.figure(figsize=(12, 8))
plt.axvline(x=0, color='red', lw=1, ls='--', alpha=0.8)
plt.axhline(y=df_2019['Average Temperature °C'].median(), color='red', lw=1, ls='--', alpha=0.8)
sns.scatterplot(data=df_2019[df_2019['total_emission'] <= 0], x='total_emission', y='Average Temperature °C')
plt.title('Outliers: Emission <= 0 vs Temperature (2019)')
plt.grid(True)
plt.show()

# Display area, total emission, and temperature for emission <= 0 (2019)
print(df_2019[df_2019['total_emission'] <= 0][['Area', 'total_emission', 'Average Temperature °C']], "\n")

# View 2020 data - Drop NA
df_2020 = df[df['Year'] == 2020].dropna(axis=1)
df_2020.info()

# View 2020 data - Reset index
df_2020 = df[df['Year'] == 2020].reset_index(drop=True)
print(df_2020.head(), "\n")

# Drop Area and Year for clustering
df_clus = df_2020.drop(['Area', 'Year'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_clus_imputed = imputer.fit_transform(df_clus)

# Standardize data
scaler = StandardScaler()
df_sc = pd.DataFrame(scaler.fit_transform(df_clus_imputed), columns=df_clus.columns)

# Elbow Method to find optimal clusters
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(df_sc)
    distortions.append(km.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.grid(True)
plt.show()

# Apply KMeans clustering
model = KMeans(n_clusters=4, random_state=1, n_init=10)
model.fit(df_sc)
df_clus['Country'] = df_2020['Area']
df_clus['Cluster'] = model.labels_

# Plot cluster counts
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df_clus.groupby('Cluster')['Country'].count().plot.bar(color='skyblue')
plt.title('Countries per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df_clus.groupby('Cluster')['Country'].count().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Cluster Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Bar-style summary of cluster means
print("\nCluster-wise Mean Summary:\n")
print(df_clus.groupby('Cluster').mean(numeric_only=True).style.bar(axis=0))

# Cluster-wise boxplot visualization
clus_col = [
    'Rice Cultivation', 'Drained organic soils (CO2)', 'Pesticides Manufacturing',
    'Food Transport', 'Food Retail', 'On-farm Electricity Use', 'Food Packaging',
    'Agrifood Systems Waste Disposal', 'Food Processing', 'Fertilizers Manufacturing',
    'Manure left on Pasture', 'Fires in organic soils', 'Rural population',
    'Urban population', 'Total Population - Male', 'Total Population - Female',
    'Average Temperature °C'
]

plt.figure(figsize=(20, 20))
for i, col in enumerate(clus_col):
    plt.subplot(8, 4, i + 1)
    sns.boxplot(data=df_clus, y=col, x='Cluster', palette='Set3')
    plt.title(col, fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()

# Total emission column should be in df_clus for the scatterplot
# If not already there, compute it as the sum of relevant columns
if 'total_emission' not in df_clus.columns:
    df_clus['total_emission'] = df_clus[clus_col].sum(axis=1)

# Cluster-wise scatterplot vs total emission
plt.figure(figsize=(20, 20))
for i, col in enumerate(clus_col):
    plt.subplot(8, 4, i + 1)
    sns.scatterplot(data=df_clus, x=col, y='total_emission', hue='Cluster', palette='Pastel1')
    plt.title(col, fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()

# PCA Transformation
pca = PCA(n_components=2, random_state=1)
pca_features = pca.fit_transform(df_sc)
df_clus['PCA1'] = pca_features[:, 0]
df_clus['PCA2'] = pca_features[:, 1]

# Mean of PCA components by cluster
print("\nMean PCA Components by Cluster:\n")
print(df_clus.groupby('Cluster')[['PCA1', 'PCA2']].mean().T.style.bar(axis=1))

# Display PCA components
pca_df = pd.DataFrame(pca.components_, columns=df_sc.columns, index=['PCA1', 'PCA2'])
print("\nPCA Components Matrix:\n")
print(pca_df.T)

# PCA Scatter plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
scatter = ax.scatter(df_clus['PCA1'], df_clus['PCA2'], c=df_clus['Cluster'], cmap='tab10', alpha=0.8)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3'],
           title='Cluster', loc='upper left', bbox_to_anchor=(1, 1))
plt.title('PCA Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

# Filter and display specific clusters
print("\nFiltered Clusters (1, 2, 3):\n")
print(df_clus[df_clus['Cluster'].isin([1, 2, 3])][['Country', 'Cluster', 'PCA1', 'PCA2']])